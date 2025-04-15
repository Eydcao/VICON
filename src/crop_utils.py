import torch
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import enum


class NODE_TYPE(enum.Enum):
    """
    Enumeration for node types in a grid.
    """

    INTERIOR = 0
    BOUNDARY = 1


def create_grid(starting_points, grid_size):
    """
    Create a grid based on starting points and length vectors.

    Parameters:
    starting_points (torch.Tensor): A tensor of shape [..., 2] where the last dimension contains starting_x and starting_y.
    grid_size (list or tuple): [num_gridx, num_gridy].

    Returns:
    torch.Tensor: A grid of shape [..., num_gridx, num_gridy, 2] containing the grid coordinates.
    """
    dxs = torch.arange(grid_size[0]).to(starting_points.device)
    dys = torch.arange(grid_size[1]).to(starting_points.device)

    dx_grid, dy_grid = torch.meshgrid(dxs, dys, indexing="ij")

    starting_x = starting_points[..., 0, None, None]
    starting_y = starting_points[..., 1, None, None]

    grid_x = starting_x + dx_grid
    grid_y = starting_y + dy_grid

    grid = torch.stack([grid_x, grid_y], dim=-1)

    return grid


def apply_modulo(grid, bound):
    """
    Apply modulo operation on the input grid to wrap around the boundaries.

    Parameters:
    grid (torch.Tensor): A tensor of shape [..., num_gridx, num_gridy, dim].
    bound (list or tuple): A list or tuple containing [xmax, ymax].

    Returns:
    torch.Tensor: A tensor where the last two dimensions have been modded by xmax and ymax, respectively.
    """
    gridx = grid[..., 0]
    gridy = grid[..., 1]

    modx = torch.remainder(gridx, bound[0])
    mody = torch.remainder(gridy, bound[1])

    mod_data = torch.stack((modx, mody), dim=-1)

    return mod_data


def sample_data(data, grid):
    """
    Subsample the data tensor based on the provided indices.

    Parameters:
    data (torch.Tensor): A tensor of shape [f, c, nx, ny].
    grid (torch.Tensor): A tensor of indices of shape [f, gridx_num, gridy_num, 2], where 2 corresponds to (x, y).

    Returns:
    torch.Tensor: The sampled tensor using the input grid x, y ids of shape [f, c, gridx_num, gridy_num].
    """
    f, gridx_num, gridy_num, dim = grid.shape
    c = data.shape[1]

    # Extract x and y indices from the grid
    x_indices = grid[:, :, :, 0]
    y_indices = grid[:, :, :, 1]

    # Prepare indices for advanced indexing
    batch_indices = torch.arange(f).view(f, 1, 1).expand(-1, gridx_num, gridy_num)

    # Use advanced indexing to gather the data
    sampled_data = data[batch_indices, :, x_indices, y_indices].permute(0, 3, 1, 2).contiguous()

    return sampled_data


def crop_data_with_boundary_treatment(frames, crop_start_loc, crop_len, mode):
    """
    Crop data with boundary treatment.

    Parameters:
    frames (torch.Tensor): A tensor containing the frames of shape [f, c, nx, ny].
    crop_start_loc (torch.Tensor): The starting location for cropping of shape [f, 2].
    crop_len (list or tuple): The length of the crop in x and y directions [dx, dy].
    mode (str): The boundary treatment mode, either 'padding' or 'periodic'.

    Returns:
    torch.Tensor: The cropped data tensor.
    """
    crop_grid = create_grid(crop_start_loc, crop_len)  # (f, gridx_num, gridy_num, 2)
    # print("crop_grid", crop_grid.device)

    if mode == "padding":
        nx, ny = frames.shape[-2], frames.shape[-1]
        mask_outside = torch.logical_or(
            crop_grid < 0, crop_grid >= frames.new_tensor([nx, ny])
        )  # (f, gridx_num, gridy_num, 2)
        mask_outside = torch.any(mask_outside, dim=-1)  # (f, gridx_num, gridy_num)
        crop_grid[mask_outside] = 0
        data = (
            sample_data(frames, crop_grid).permute(0, 2, 3, 1).contiguous()
        )  # (f, c, gridx_num, gridy_num) -> (f, gridx_num, gridy_num, c)
        c_dim = data.shape[-1]
        pad_info = frames.new_zeros(c_dim, dtype=torch.float32)
        pad_info[-1] = NODE_TYPE.BOUNDARY.value
        mask_outside = mask_outside.unsqueeze(-1)  # (f, gridx_num, gridy_num, 1)
        data = (
            torch.where(mask_outside == 1, pad_info, data).permute(0, 3, 1, 2).contiguous()
        )  # (f, gridx_num, gridy_num, c) -> (f, c, gridx_num, gridy_num)
    elif mode == "periodic":
        crop_grid = apply_modulo(crop_grid, frames.shape[-2:])  # (f, gridx_num, gridy_num, 2)
        data = sample_data(frames, crop_grid)  # (f, c, gridx_num, gridy_num)

    return data


def crop_frames(frames, crop_center, crop_len, pad_mode):
    """
    Crop frames with specified boundary treatment.

    Parameters:
    frames (torch.Tensor): A tensor containing the frames of shape [f, c, nx, ny].
    crop_center (torch.Tensor): The center location for cropping of shape [f, 2].
    crop_len (list or tuple): The length of the crop in x and y directions [dx, dy].
    pad_mode (str): The boundary treatment mode, either 'padding' or 'periodic'.

    Returns:
    torch.Tensor: The cropped frames.
    """
    crop_left_lower = crop_center - crop_len // 2
    # check devices
    # print(frames.device)
    # print(crop_left_lower.device)
    # print(crop_len.device)
    cropped_data = crop_data_with_boundary_treatment(frames, crop_left_lower, crop_len, pad_mode)
    return cropped_data
