import wandb
import random
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pytz
import utils
from dataset import all_datasets
import models
from trainer import Trainer
from datetime import datetime
from rollout_utils import rollout_one_traj, restore_model, plot_traj
from rollout_strategy import gen_strategy_list, print_strategy
from tqdm import tqdm

# # In rollout somehow the global dev is defualt to cpu
# # NOTE correct it to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_error(residuals, tar, error_cfg):
    """
    Input shape: (t, c, h, w)
    Output shape: (t, c)
    """
    error = torch.sqrt((residuals**2).flatten(2).mean(2))  # (t, c)
    if error_cfg.scheme == "scale_by_size":
        scale = torch.sqrt((tar**2).flatten(2).mean(2))
        scale = torch.where(torch.abs(scale) < error_cfg.eps, 1, scale)
        error = torch.div(error, scale)
    elif error_cfg.scheme == "scale_by_std":
        scale = torch.std(tar.flatten(2), dim=2)  # (t, c)
        scale = torch.where(scale < error_cfg.eps, error_cfg.eps, scale)
        error = torch.div(error, scale)
    elif error_cfg.scheme == "abs":
        error = error  # (t, c) no scaling
    else:
        raise ValueError(f"Unknown error scheme: {error_cfg.scheme}")
    return error


def save_results(results, ground_truth, folder, name):
    # results: [t, c, h, w], ground_truth: [t, c, h, w]
    # save as npz
    os.makedirs(f"./results_rollout/{folder}", exist_ok=True)
    np.savez(f"./results_rollout/{folder}/{name}.npz", results=results, ground_truth=ground_truth)


@torch.no_grad()
def run_rollout(cfg):
    utils.set_seed(cfg.train_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.train_seed)

    print(OmegaConf.to_yaml(cfg))

    if cfg.board:
        wandb_run = wandb.init(
            project="rollout",
            entity="liuyang-research",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb_run = None

    # Creation of instances
    if cfg.model.type == "crop":
        model = models.ICON_CROPPED(cfg.model)
        print("Using cropped (new) model")
    elif cfg.model.type == "nocrop":
        model = models.ICON_UNCROPPED(cfg.model)
        print("Using uncropped (ancient) model")
    else:
        raise ValueError("Unknown model type: {}".format(cfg.model.type))

    # create model
    # ckpt_path = f"{FLAGS.ckpt_dir}/{FLAGS.ckpt_stamp}.pth"
    ckpt_path = os.path.join(cfg.rollout.ckpt_dir, cfg.rollout.ckpt_stamp + ".pth")
    restore_model(model, ckpt_path)
    # create dataset
    rollout_datasets = all_datasets(cfg.datasets, 0, cfg.rollout_seed, "rollout")

    # create trainer
    trainer = Trainer(model, cfg.model, cfg.opt, cfg.loss, trainable_mode=cfg.trainable_mode, amp=cfg.amp)
    # create data loaders
    rollout_loaders = {
        k: torch.utils.data.DataLoader(v, batch_size=1, num_workers=0, pin_memory=True)
        for k, v in rollout_datasets.items()
    }

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    stamp = time_stamp
    print("stamp: {}".format(stamp))

    # Initialize error accumulator dictionary
    errors_all = {}

    # Iterate through data
    for dataset_type, rollout_loader in rollout_loaders.items():
        # # For test only remove later NOTE
        batch_cnt = 0
        errors_per_type = []
        for batch in tqdm(rollout_loader):
            batch_cnt += 1
            # if batch_cnt > 2:  # For test only remove later NOTE
            #     break
            # b_traj: [bs=1, t, c, h, w]
            # b_c_mask: [bs, c]
            _, b_traj, b_c_mask = batch
            # # to device
            b_traj = b_traj.to(trainer.device)
            b_c_mask = b_c_mask.to(trainer.device)
            # [t, c, h, w]
            full_frame = b_traj[0]  # [t, c, h, w]
            c_mask = b_c_mask[0]

            # Generate strategy list
            T_len = full_frame.shape[0]
            gt_ref_steps = cfg.rollout.gt_ref_steps
            strategy_list = gen_strategy_list(cfg.rollout, T_len)

            # Prepare the pred results, while the first gt_ref_steps+1 are the same as the full_frame
            results = full_frame.new_zeros(full_frame.shape)
            results[:gt_ref_steps, ...] = full_frame[:gt_ref_steps, ...]

            # Rollout
            results = rollout_one_traj(
                trainer, dataset_type, cfg, results, c_mask, tc_rng, strategy_list
            )  # [t, c, h, w]

            # Compute the mean squared error for the current trajectory
            # [t,c,h,w] -> [t, c]

            traj_error = get_error(full_frame - results, full_frame, cfg.rollout.error)  # [t, c]

            # get meaningful channel in traj_error only
            valid_c_idx = torch.where(c_mask == 1)[0]
            traj_error = traj_error[:, valid_c_idx]
            # get temporal idx after gt_ref_steps + 1 only, as previous frames are given
            traj_error = traj_error[gt_ref_steps:, :]  # [t, c]

            errors_per_type.append(traj_error)

            if batch_cnt <= cfg.rollout.save:  # if save > 0, save the results and early stop
                save_results(
                    results.numpy(force=True),
                    full_frame.numpy(force=True),
                    folder=f"{cfg.rollout.ckpt_dir.split('/')[-1]}_{cfg.rollout.strategy}",
                    name=f"{dataset_type}_case{batch_cnt}",
                )
                if batch_cnt == cfg.rollout.save:
                    break  # only eval and save the first cfg.rollout.save trajectories

            if batch_cnt <= 2:  # only plot the first 2 trajectories
                print("")
                print(dataset_type)
                print("strategy_list:")
                print_strategy(strategy_list)
                print("valid c idx:", valid_c_idx)
                print("pred traj error shape:", traj_error.shape)

                if cfg.rollout.save <= 0:  # if save > 0, skip plotting, since we already saved the results
                    plot_traj(
                        f"case{batch_cnt}",
                        dataset_type,
                        results.numpy(force=True),
                        full_frame.numpy(force=True),
                        wandb_run,
                        local=False,
                        upload=cfg.board,
                    )

        errors_all[dataset_type] = torch.stack(errors_per_type)  # [dataset_size, t, c]

    # Post-processing: compute the overall statistics for each dataset type
    print("\n-------------- Printing error averaged over time and channel --------------\n")
    for dataset_type, errors in errors_all.items():
        error_avg = torch.mean(errors, dim=(1, 2))  # (dataset_size, )
        std_error, mean_error = torch.std_mean(error_avg)

        print(f"Dataset Type: {dataset_type} (size: {errors.size(0)})")
        print(f"Mean error:   {mean_error.item():.6f}")
        print(f"Std error:    {std_error.item():.6f}")
        print()
    print("\n\n")

    print("-------------- Printing error for each channel --------------\n")
    for dataset_type, errors in errors_all.items():
        error_avg_over_time = torch.mean(errors, dim=1)  # (dataset_size, c)
        std_error, mean_error = torch.std_mean(error_avg_over_time, dim=0)

        print(f"Dataset Type: {dataset_type}")
        print(f"Mean error:   {mean_error.tolist()}")
        print(f"Std error:    {std_error.tolist()}")
        print()
    print("\n\n")

    print("-------------- Printing error for each time step --------------\n")
    for dataset_type, errors in errors_all.items():
        # [1, size_of_error_metric]
        error_avg_over_chan = torch.mean(errors, dim=2)  # (dataset_size, t)
        std_error, mean_error = torch.std_mean(error_avg_over_chan, dim=0)

        print(f"Dataset Type: {dataset_type}")
        # print(f"Mean error:   {mean_error}")
        # print(f"Std error:    {std_error}")
        # record 1 5 10 -1 steps error, if the length is enough
        T_len = mean_error.shape[0]
        if T_len >= 11:
            print(f"Mean error at 1, 5, 10, last steps: {mean_error[[0, 4, 9, -1]].tolist()}")
            print(f"Std error at 1, 5, 10, last steps:  {std_error[[0, 4, 9, -1]].tolist()}")
        elif T_len >= 6:
            print(f"Mean error at 1, 5, last steps:     {mean_error[[0, 4, -1]].tolist()}")
            print(f"Std error at 1, 5, last steps:      {std_error[[0, 4, -1]].tolist()}")
        elif T_len >= 2:
            print(f"Mean error at 1, last steps:        {mean_error[[0, -1]].tolist()}")
            print(f"Std error at 1, last steps:         {std_error[[0, -1]].tolist()}")
        else:
            print("There is only one time step, no statistics to show.")
        print()

    print("\n\n")
    for dataset_type, errors in errors_all.items():
        # [1, size_of_error_metric]
        error_avg_over_chan = torch.mean(errors, dim=2)  # (dataset_size, t)
        std_error, mean_error = torch.std_mean(error_avg_over_chan, dim=0)

        print(f"Dataset Type: {dataset_type}\n")
        T_len = mean_error.shape[0]
        print("Step\tError Mean\tError Std")
        for i in range(T_len):
            print(f"{i+1}\t{mean_error[i]:.6f}\t{std_error[i]:.6f}")
            if cfg.board:
                wandb.log(
                    {
                        f"{dataset_type}_step": i + 1,
                        f"{dataset_type}_error_mean": mean_error[i],
                        f"{dataset_type}_error_std": std_error[i],
                    }
                )

        print("\n\n")


@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    run_rollout(cfg)


if __name__ == "__main__":
    main()
