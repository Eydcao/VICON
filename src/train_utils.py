from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import make_image, merge_images, plt_to_wandb
import wandb


def cross_batch_process(pairs, tc_rng, scheme):
    """
    pairs: (bs, pairs, 2, c, x1, ..., xd)
    """
    if scheme == "random":
        bs, ps, *shape = pairs.shape
        pairs = pairs.view(-1, *pairs.shape[2:])  # (bs * pairs, 2, c, x1, ..., xd)
        idx = torch.randperm(pairs.shape[0], generator=tc_rng)
        pairs = pairs[idx]
        pairs = pairs.view(bs, ps, *pairs.shape[1:])  # (bs, pairs, 2, c, x1, ..., xd)
        return pairs
    elif scheme == "no":
        return pairs
    else:
        raise ValueError("scheme {} not supported".format(scheme))


def board_loss(trainer, pairs, prefix, cfg):
    loss = trainer.get_loss(pairs)
    print(f"train step: {trainer.train_step}, {prefix}_loss: {loss}")
    if cfg.board:
        wandb.log({"step": trainer.train_step, f"{prefix}_loss": loss})


def print_error(types, trainer, pairs, prefix):
    # print error
    error_mean_dict, error_std_dict = trainer.get_error(types, pairs)
    # for each key in error_mean_dict, error_std_dict, print the mean and std
    for key in error_mean_dict:
        error_mean, error_std = error_mean_dict[key], error_std_dict[key]
        x = np.arange(len(error_mean))
        list_elements = [x]
        headers = [f"{prefix} {key}"]
        for cid in range(error_mean.shape[1]):
            list_elements.append(error_mean[:, cid])
            list_elements.append(error_std[:, cid])
            headers.append(f"emean, c:{cid}")
            headers.append(f"estd, c:{cid}")
        table = list(zip(*list_elements))
        # transpose = list(zip(*table))
        print(tabulate(table, headers=headers, tablefmt="grid"))


def _get_label(data):
    """
    Extract the label data from the input data.
    """
    if isinstance(data, (list, tuple)):
        # Assuming data is a tuple or list of (init_data, end_data)
        cond, label = data[0], data[1]  # data[2] could be c_mask
    else:
        # Assuming data is a single tensor of shape [bs, pairs, 2, c, h, w]
        cond = data[:, :, 0]
        label = data[:, :, 1]
    return cond, label


def eval_plot(pde_types, trainer, pairs, prefix, in_idx, out_idx, bid, cfg):
    pde_type = pde_types[bid]
    pred, _ = trainer.get_pred(pairs)
    pred = pred.detach().cpu().numpy()  # (bs, ps, co, he, we)
    cond, label = _get_label(pairs)
    cond = cond.detach().cpu().numpy()  # (bs, ps, ci, hi, wi), input
    label = label.detach().cpu().numpy()  # (bs, ps, ci, he, we)
    # remove the type channel
    cond = cond[:, :, :-1]
    label = label[:, :, :-1]
    diff = pred - label  # (bs, ps, c, h, w)
    bs, ps, c, h, w = diff.shape
    plot_dict = {}
    figs = [[None for i in range(4)] for j in range(c)]  # (c, 4)
    for pid in range(ps):
        # the pid here is effective pid, ie effective_pair_num = pair_num * crop_num
        # get pair id and crop id
        pair_id = pid // trainer.model_cfg.n_crops
        crop_id = pid % trainer.model_cfg.n_crops
        for cid in range(c):
            figs[cid][0] = make_image(
                cond[bid, pid, cid, :, :],
                wandb=False,
                title=f"step:{trainer.train_step},type:{pde_type},ex:{pid},idx:{in_idx[bid,pair_id]},crop:{crop_id},c:{cid},input",
            )
            figs[cid][1] = make_image(
                label[bid, pid, cid, :, :],
                wandb=False,
                title=f"step:{trainer.train_step},type:{pde_type},ex:{pid},idx:{out_idx[bid,pair_id]},crop:{crop_id},c:{cid},label",
            )
            figs[cid][2] = make_image(
                pred[bid, pid, cid, :, :],
                wandb=False,
                title=f"step:{trainer.train_step},type:{pde_type},ex:{pid},crop:{crop_id},c:{cid},pred",
            )
            figs[cid][3] = make_image(
                diff[bid, pid, cid, :, :],
                wandb=False,
                title=f"step:{trainer.train_step},type:{pde_type},ex:{pid},crop:{crop_id},c:{cid},diff",
            )
        merged_image = merge_images(figs, spacing=0)
        plot_dict[f"{prefix}_plot_ex{pid}"] = plt_to_wandb(merged_image, cfg={"caption": f"ex:{pid}"})
        plt.close("all")
    if cfg.board:
        wandb.log({"step": trainer.train_step, **plot_dict})


def get_data_from_looper(looper_per_type, tc_rng, cfg):
    type_list = []
    pairs_list = []
    in_idx_list = []
    out_idx_list = []
    for type_idx, (_, looper) in enumerate(looper_per_type.items()):
        tmp_types, tmp_pairs, tmp_in_idx, tmp_out_idx, _ = next(looper)
        type_list.append(tmp_types)
        pairs_list.append(tmp_pairs)
        in_idx_list.append(tmp_in_idx)
        out_idx_list.append(tmp_out_idx)

    t_types = sum(type_list, [])
    t_pairs = tuple(torch.cat([pair[i] for pair in pairs_list], dim=0) for i in range(len(pairs_list[0])))
    t_in_idx = torch.cat(in_idx_list, dim=0)
    t_out_idx = torch.cat(out_idx_list, dim=0)

    t_pairs = cross_batch_process(t_pairs, tc_rng, cfg.data_cross_batch)  # actually no cross_batch process for now

    return type_list, pairs_list, in_idx_list, out_idx_list, t_types, t_pairs, t_in_idx, t_out_idx
