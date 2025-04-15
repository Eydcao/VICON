from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import make_image, merge_images, plt_to_wandb
import wandb
from rollout_strategy import fix_single_step_strategy
from dataset_utils import crop_frames_pairs
from crop_utils import crop_frames
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class ErrorAccumulator:
#     def __init__(self, size, device="cuda"):
#         self.normalizer = Normalizer(size=size, device=device)

#     def accumulate(self, traj_loss):
#         self.normalizer(traj_loss, accumulate=True)

#     def get_statistics(self):
#         mean = self.normalizer._mean()
#         std = self.normalizer._std_with_epsilon()
#         return mean, std


def restore_model(model, ckpt_path):
    # Model restoration from the last checkpoint in store_dir
    model = model.module if hasattr(model, "module") else model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("restored params from {}".format(ckpt_path))


def uniform_patchify(frame, p_res, crop_res, pad_mode):
    # frame: [f=1, c, h, w]
    # p_res: patch resolution used for determining patch centers
    # crop_res: used to determine how big the patch is

    # return: [f, num_uniform_crops, c, h_in, w_in]

    # Extract the dimensions of the input frame
    f_len, c, h, w = frame.shape  # [f, c, h, w]
    crop_nx = h // p_res  # Number of patches along the x
    crop_ny = w // p_res  # Number of patches along the y

    res = []
    for ix in range(crop_nx):
        for iy in range(crop_ny):
            # Calculate the center of the patch
            cx = (2 * ix + 1) * p_res // 2
            cy = (2 * iy + 1) * p_res // 2
            # (f, 2)
            crop_center = torch.tensor([cx, cy]).unsqueeze(0).repeat(f_len, 1).to(device)
            # Crop the patch
            # [f, c, h_in, w_in]
            patch = crop_frames(frame, crop_center, torch.tensor([crop_res, crop_res]).to(device), pad_mode)

            res.append(patch)

    # to torch tensor
    # [num_uniform_crops, f=1, c, h_in, w_in]
    res = torch.stack(res, dim=0)

    return res


def uniform_unpatchify(pred, p_res, h, w):
    # pred [crop_nx * crop_ny, c, ph, pw]

    crop_nx = h // p_res  # Number of patches along the x
    crop_ny = w // p_res  # Number of patches along the y

    # -> [crop_nx, crop_ny, c, ph, pw]
    pred = pred.view(crop_nx, crop_ny, *pred.shape[1:])
    # -> [c, crop_nx, ph, crop_ny, pw]
    pred = pred.permute(2, 0, 3, 1, 4)
    # -> [c, crop_nx * ph = h, crop_ny * pw = w]
    pred = pred.reshape(pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3] * pred.shape[4])
    return pred


def pred(trainer, dataset_type, cfg, full_demo_cond, full_demo_qoi, full_quest_cond, c_mask, tc_rng):
    """
    full_demo_cond: [rollout_demo_num, c, h, w],
    full_demo_qoi: [rollout_demo_num, c, h, w],
    full_quest_cond: [1, c, h, w]
    c_mask: [c]

    return: # [c-1, h, w]
    """
    # get split_scheme
    split_scheme = cfg.datasets.split_scheme
    if split_scheme == "crop_split":
        # -> correct shapes

        # [nx*ny, 1, c, h_out, w_out]
        quest_cond = uniform_patchify(
            full_quest_cond,
            cfg.datasets.crop_len_out,
            cfg.datasets.crop_len_in,
            cfg.datasets.types[dataset_type].pad_mode,
        )

        # for every frame of full_demo_cond, full_demo_qoi
        # get a random center and crop the patch, respectively
        # [rollout_demo_num, c, h_in, w_in], [rollout_demo_num, c, h_out, w_out]
        cropped_demo_cond, cropped_demo_qoi = crop_frames_pairs(
            full_demo_cond,
            full_demo_qoi,
            tc_rng,
            cfg.datasets.crop_len_in,
            cfg.datasets.crop_len_out,
            cfg.datasets.types[dataset_type].pad_mode,
        )
        # ->[1, rollout_demo_num, c, h_in, w_in] -> [nx*ny, rollout_demo_num, c, h_in, w_in]
        demo_cond = cropped_demo_cond.unsqueeze(0).repeat(quest_cond.shape[0], 1, 1, 1, 1)
        # ->[1, rollout_demo_num, c, h_out, w_out] -> [nx*ny, rollout_demo_num, c, h_out, w_out]
        demo_qoi = cropped_demo_qoi.unsqueeze(0).repeat(quest_cond.shape[0], 1, 1, 1, 1)
        # form the full sequence
        # dummy in [nx*ny, 1, c, h_out, w_out]
        quest_qoi = demo_qoi.new_zeros(
            (quest_cond.shape[0], 1, demo_qoi.shape[2], demo_qoi.shape[3], demo_qoi.shape[4])
        )

        # [nx*ny, pairs=rollout_demo_num+1, c, h_in, w_in]
        cond = torch.cat([demo_cond, quest_cond], dim=1)
        # [nx*ny, pairs=rollout_demo_num+1, c, h_out, w_out]

        qoi = torch.cat([demo_qoi, quest_qoi], dim=1)
        # c_mask: (c) -> (bs=nx*ny, c)
        c_mask = c_mask.unsqueeze(0).repeat(cond.shape[0], 1)
    elif split_scheme == "split":
        demo_cond, demo_qoi = full_demo_cond, full_demo_qoi
        quest_cond = full_quest_cond
        # -> correct shapes
        # shape of cond=(bs=1, pairs=rollout_demo_num+1, c, h_init, w_init)
        # shape of qoi=(bs=1, pairs, c, h_end, w_end)
        # shape of c_mask=(bs=1, c)
        demo_cond = demo_cond.unsqueeze(0)  # (1, rollout_demo_num, c, h_in, w_in)
        demo_qoi = demo_qoi.unsqueeze(0)  # (1, rollout_demo_num, c, h_out, w_out)
        c_mask = c_mask.unsqueeze(0)  # (1, c)
        quest_cond = quest_cond.unsqueeze(0)  # (1, 1, c, h_in, w_in)
        # form the full sequence
        cond = torch.cat([demo_cond, quest_cond], dim=1)  # [1, pairs, c, h_in, w_in]
        qoi = torch.cat([demo_qoi, torch.zeros_like(quest_cond)], dim=1)  # [1, pairs, c, h_in, w_in]
    else:
        raise ValueError("split_scheme {} not supported in rollout".format(split_scheme))

    # get the prediction
    pred, _ = trainer.get_pred((cond, qoi, c_mask))
    # pred: [bs=1 or nx * ny, pairs=rollout_demo_num+1, c-1, h_out, w_out]
    # c-1 since the last channel is the type channel, thus removed

    # get the last one in dim:pairs, which is the prediction
    quest_qoi_pred = pred[:, -1, ...]  # [bs=1 or nx * ny, c-1, h_out, w_out]
    if split_scheme == "crop_split":
        out = uniform_unpatchify(
            quest_qoi_pred, cfg.datasets.crop_len_out, full_quest_cond.shape[-2], full_quest_cond.shape[-1]
        )
    elif split_scheme == "split":
        out = quest_qoi_pred[0]  # [c-1, h, w]
    else:
        raise ValueError("split_scheme {} not supported in rollout".format(split_scheme))
    return out


def rollout_one_traj(trainer, dataset_type, cfg, results, c_mask, tc_rng, strategy_list):
    """
    NOTE: never pass the ground truth to prediction function!
    results: the results sequence with some init frame assigned, [t, c, h, w]
             NOTE while the remaining frames are zero, the type channel is not zero
    return: the results with full prediction, [t, c, h, w]
    """
    # generate strategy list
    # check devices
    # print("traner device", trainer.device)
    # print("results device", results.device)
    # print("c_mask device", c_mask.device)

    # NOTE assume type channle is static, record it
    bc_type = results[0, -1, :, :].clone()

    for strategy in strategy_list:
        demo_cond_idxs, demo_qoi_idxs, quest_cond_idx, quest_qoi_idx = strategy
        demo_cond = results[demo_cond_idxs]
        demo_qoi = results[demo_qoi_idxs]
        quest_cond = results[[quest_cond_idx]]
        out = pred(trainer, dataset_type, cfg, demo_cond, demo_qoi, quest_cond, c_mask, tc_rng)
        results[quest_qoi_idx, :-1, :, :] = out  # the last channel is the type channel
        # set it up correctly in case this frame is used in the future
        results[quest_qoi_idx, -1, :, :] = bc_type

    return results


def plot_traj(prefix, dataset_type, traj_pred, traj_gt, wandb_run, local, upload):
    """
    compact version of eval_plot, try to put all in one plot
    traj [t, c, h, w]
    prefix and dataset_type are for naming
    """
    t, c, h, w = traj_pred.shape
    figs = [[None for i in range(c * 3)] for j in range(t)]  # (t, c * 3), 3 for pred, gt, diff
    for tid in range(t):
        for cid in range(c):
            figs[tid][cid * 3 + 0] = utils.make_image(
                traj_pred[tid, cid], wandb=False, title=f"{prefix}, type:{dataset_type}, t:{tid}, c:{cid}"
            )
            figs[tid][cid * 3 + 1] = utils.make_image(
                traj_gt[tid, cid], wandb=False, title=f"{prefix}, type:{dataset_type}, t:{tid}, c:{cid}"
            )
            figs[tid][cid * 3 + 2] = utils.make_image(
                traj_pred[tid, cid] - traj_gt[tid, cid],
                wandb=False,
                title=f"{prefix}, type:{dataset_type}, t:{tid}, c:{cid}",
            )
    # merge the figs
    merged_image = utils.merge_images(figs, spacing=0)

    if local:
        merged_image.save(f"{prefix}_type{dataset_type}_t{tid}_c{cid}" + ".png")

    if upload:
        fig = utils.plt_to_wandb(merged_image, cfg=None)
        wandb_run.log({f"{prefix}_type{dataset_type}_t{tid}_c{cid}": fig})

    plt.close("all")
