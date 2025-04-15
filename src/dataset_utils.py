import torch
from crop_utils import crop_frames
from omegaconf import OmegaConf
from crop_utils import NODE_TYPE


class InfiniteDataLooper:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.data_iter_num = 0

    def __next__(self):
        try:
            out = next(self.data_iter)
        except StopIteration:
            print(f"reached end of data loader, restart {self.data_iter_num}")
            self.data_iter_num += 1
            self.data_iter = iter(self.data_loader)
            out = next(self.data_iter)
        return out


def random_rotate_flip_tuple(image_tuple, tc_rng):
    """
    Apply a random rotation and flip (but the same) to the image tuple
    image_tuple: ((..., x1, x2), (..., x3, x4))
    """
    image1, image2 = image_tuple
    # Define the rotation angles

    # Randomly choose a rotation angle
    random_k = torch.randint(-2, 2, (), generator=tc_rng)  # in [-2, 2)
    # apply the same rotation and flip to both images
    image1 = torch.rot90(image1, k=random_k, dims=[-2, -1])
    image2 = torch.rot90(image2, k=random_k, dims=[-2, -1])
    # Randomly flip the image
    if torch.rand(1, generator=tc_rng).item() < 0.5:
        image1 = torch.flip(image1, dims=[-1])
        image2 = torch.flip(image2, dims=[-1])
    if torch.rand(1, generator=tc_rng).item() < 0.5:
        image1 = torch.flip(image1, dims=[-2])
        image2 = torch.flip(image2, dims=[-2])

    image_tuple = (image1, image2)
    return image_tuple


def rotate_flip_augmentation(pairs, tc_rng, rotate_flip):
    if rotate_flip == "none":
        pass
    elif rotate_flip == "pairwise":
        new_crop_data_in = []
        new_crop_data_out = []
        for pid in range(crop_data_in.shape[0]):
            tmp_rotated_in, tmp_rotated_out = random_rotate_flip_tuple((crop_data_in[pid], crop_data_out[pid]), tc_rng)
            new_crop_data_in.append(tmp_rotated_in)
            new_crop_data_out.append(tmp_rotated_out)
        crop_data_in = torch.stack(new_crop_data_in, dim=0)
        crop_data_out = torch.stack(new_crop_data_out, dim=0)
        pairs = (crop_data_in.contiguous(), crop_data_out.contiguous())
    elif rotate_flip == "sequencewise":
        crop_data_in, crop_data_out = random_rotate_flip_tuple((crop_data_in, crop_data_out), tc_rng)
        pairs = (crop_data_in.contiguous(), crop_data_out.contiguous())
    else:
        raise ValueError("rotate_flip {} not supported".format(rotate_flip))
    return pairs


def split_data(raw_data, rng, tc_rng, cfg):
    """
    Split raw data into input-output pairs based on time indices.

    Parameters:
    raw_data (torch.Tensor): The raw data tensor of shape [t, c, x1, ..., xd].
    rng (random.Random): A random number generator instance.
    tc_rng (torch.Generator): A PyTorch random number generator instance.
    cfg (object): Configuration object containing parameters:
        - delta_t_range (tuple): Range for delta_t as (min, max).
        - demo_num (int): Number of pairs to generate.
        - monotonic (bool): If True, ensures time indices are sorted.
        - rotate_flip (bool): If True, applies rotation and flip augmentation.

    Returns:
    tuple: A tuple containing:
        - pairs (tuple): A tuple of input and output data tensors.
        - t_in (torch.Tensor): Input time indices of shape [demo_num].
        - t_out (torch.Tensor): Output time indices of shape [demo_num].
        - delta_t (torch.Tensor): Time difference between input and output frames.
    """
    t = raw_data.shape[0]
    delta_t = torch.randint(
        cfg.delta_t_range[0], cfg.delta_t_range[1] + 1, (), generator=tc_rng
    )  # in [delta_t_min, delta_t_max]
    t_in = torch.randint(0, t - delta_t, (cfg.demo_num,), generator=tc_rng)  # in [0, t - delta_t - 1]

    if cfg.monotonic:
        t_in, _ = torch.sort(t_in)

    t_out = t_in + delta_t
    data_in = raw_data[t_in]  # (demo_num, c, x1, ..., xd)
    data_out = raw_data[t_out]  # (demo_num, c, x1, ..., xd)
    # # to (demo_num * c, ...)
    # data_in = data_in.reshape(-1, *data_in.shape[2:])
    # data_out = data_out.reshape(-1, *data_out.shape[2:])

    # Apply augmentation such as rotation and flip
    pairs = (data_in, data_out)
    pairs = rotate_flip_augmentation(pairs, tc_rng, cfg.rotate_flip)

    return pairs, t_in, t_out, delta_t


def crop_frames_pairs(frames_in, frames_out, tc_rng, crop_len_in, crop_len_out, pad_mode):
    """
    Function:
    Crop the input and output frames based on random but consistent center and with crop lengthes, respectively.

    Parameters:
    frames_in: [t, c, x, y]
    frames_out: [t, c, x, y]
    crop_len_in: int
    crop_len_out: int
    pad_mode: str

    Returns:
    tuple: A tuple containing:
        - crop_data_in: [t, c, crop_len_in, crop_len_in]
        - crop_data_out: [t, c, crop_len_out, crop_len_out]
    """
    # Generate crop center for the input and output frames
    nx, ny = frames_in.shape[-2], frames_in.shape[-1]
    # check devs
    # print(frames_in.device)
    # print(frames_out.device)

    crop_xc = torch.randint(0, nx, (frames_in.shape[0],), generator=tc_rng).to(frames_in.device)
    crop_yc = torch.randint(0, ny, (frames_in.shape[0],), generator=tc_rng).to(frames_in.device)
    crop_center = torch.stack((crop_xc, crop_yc), dim=-1)  # (demo_num, 2)

    # Crop the frames given center and crop length
    crop_data_in = crop_frames(
        frames_in, crop_center, torch.tensor([crop_len_in, crop_len_in], device=frames_in.device), pad_mode
    )
    crop_data_out = crop_frames(
        frames_out, crop_center, torch.tensor([crop_len_out, crop_len_out], device=frames_out.device), pad_mode
    )

    return crop_data_in, crop_data_out


def crop_split_data(raw_data, tc_rng, cfg, pad_mode):
    """
    Split raw data into input-output pairs with cropping and apply augmentations.

    Parameters:
    raw_data (torch.Tensor): The raw data tensor of shape [t, c, x, y].
    tc_rng (torch.Generator): A PyTorch random number generator instance.
    cfg (object): Configuration object containing parameters:
        - delta_t_range (tuple): Range for delta_t as (min, max).
        - demo_num (int): Number of demonstrations to generate.
        - monotonic (bool): If True, ensures time indices are sorted.
        - rotate_flip (bool): If True, applies rotation and flip augmentation.
        - crop_len_in (int): Crop length for input frames.
        - crop_len_out (int): Crop length for output frames.
    pad_mode (str): Padding mode, either 'padding' or 'periodic'.

    Returns:
    tuple: A tuple containing:
        - pairs (tuple): A tuple of input and output cropped data tensors.
        - t_in (torch.Tensor): Input time indices of shape [demo_num].
        - t_out (torch.Tensor): Output time indices of shape [demo_num].
        - delta_t (torch.Tensor): Time difference between input and output frames.
    """
    t_len = raw_data.shape[0]
    delta_t = torch.randint(
        cfg.delta_t_range[0], cfg.delta_t_range[1] + 1, (), generator=tc_rng
    )  # in [delta_t_min, delta_t_max]
    t_in = torch.randint(0, t_len - delta_t, (cfg.demo_num,), generator=tc_rng)  # in [0, t - delta_t - 1]

    if cfg.monotonic:
        t_in, _ = torch.sort(t_in)

    t_out = t_in + delta_t
    frames_in, frames_out = raw_data[t_in], raw_data[t_out]  # (demo_num, c, x, y)

    # Generate crop center for the input and output frames
    # Crop the frames given center and crop length
    crop_data_in, crop_data_out = crop_frames_pairs(
        frames_in, frames_out, tc_rng, cfg.crop_len_in, cfg.crop_len_out, pad_mode
    )

    # Apply augmentation such as rotation and flip
    pairs = (crop_data_in, crop_data_out)
    pairs = rotate_flip_augmentation(pairs, tc_rng, cfg.rotate_flip)

    return pairs, t_in, t_out, delta_t
