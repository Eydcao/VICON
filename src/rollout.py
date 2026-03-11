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
from rollout_utils import rollout_one_traj, rollout_batch_trajs, restore_model, plot_traj
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


def save_results(results, ground_truth, folder, name, err=0):
    # results: [t, c, h, w], ground_truth: [t, c, h, w]
    # save as npz
    os.makedirs(f"./results_rollout/{folder}", exist_ok=True)
    np.savez(f"./results_rollout/{folder}/{name}.npz", results=results, ground_truth=ground_truth, err=err)


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

    # Creation of model instance
    model = models.ICON_UNCROPPED(cfg.model)
    print("Using VICON model")

    # create model
    # ckpt_path = f"{FLAGS.ckpt_dir}/{FLAGS.ckpt_stamp}.pth"
    ckpt_path = os.path.join(cfg.rollout.ckpt_dir, cfg.rollout.ckpt_stamp + ".pth")
    restore_model(model, ckpt_path)
    # create dataset
    rollout_datasets = all_datasets(cfg.datasets, 0, cfg.rollout_seed, "rollout", cfg.rollout)

    # create data loaders
    batch_size = cfg.rollout.batch_size
    # create trainer - enable multi_gpu if batch_size > 1 to utilize all GPUs
    use_multi_gpu = batch_size > 1 and torch.cuda.device_count() > 1
    trainer = Trainer(model, cfg.model, cfg.opt, cfg.loss, trainable_mode=cfg.trainable_mode, amp=cfg.amp, multi_gpu=use_multi_gpu)
    print(f"Rollout batch size: {batch_size}")
    rollout_loaders = {
        k: torch.utils.data.DataLoader(v, batch_size=batch_size, num_workers=0, pin_memory=True)
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
        traj_cnt = 0  # count individual trajectories processed
        batch_cnt = 0  # count batches processed
        errors_per_type = []
        for batch in tqdm(rollout_loader):
            batch_cnt += 1
            # if batch_cnt > 2:  # For test only remove later NOTE
            #     break

            # Check if we're using dropped frames
            if cfg.rollout.dropped:
                # b_traj: [bs=1, t, c, h, w]
                # dropped_mask: [bs=1, t]
                # b_c_mask: [bs, c]
                _, b_traj, dropped_mask, b_c_mask = batch
                dropped_mask = dropped_mask[0].to(trainer.device)  # [t]
            else:
                # b_traj: [bs=1, t, c, h, w]
                # b_c_mask: [bs, c]
                _, b_traj, b_c_mask = batch
                dropped_mask = None

            current_batch_size = b_traj.shape[0]
            # to device
            b_traj = b_traj.to(trainer.device)
            b_c_mask = b_c_mask.to(trainer.device)

            # Generate strategy list (same for all trajectories in batch)
            T_len = b_traj.shape[1]
            gt_ref_steps = cfg.rollout.gt_ref_steps

            # Modify strategy generation to account for dropped frames
            strategy_list = gen_strategy_list(cfg.rollout, T_len, dropped_mask)

            if cfg.rollout.dropped:
                (strategy_list, accumulated_indices) = strategy_list
                effective_indices = [i for i in accumulated_indices if i >= gt_ref_steps]
                # sort
                effective_indices.sort()
                if len(effective_indices) == 0:
                    raise ValueError("No effective prediction found for a file.")

            if batch_size > 1:
                # Batched rollout: process all trajectories in batch simultaneously
                # Prepare the pred results for all trajectories in batch
                results_batch = b_traj.new_zeros(b_traj.shape)  # [bs, t, c, h, w]
                results_batch[:, :gt_ref_steps, ...] = b_traj[:, :gt_ref_steps, ...]

                # Rollout all trajectories in parallel
                results_batch = rollout_batch_trajs(
                    trainer, cfg, results_batch, b_c_mask, strategy_list
                )  # [bs, t, c, h, w]

                # Process each trajectory in the batch for error computation
                for i in range(current_batch_size):
                    traj_cnt += 1
                    full_frame = b_traj[i]  # [t, c, h, w]
                    results = results_batch[i]  # [t, c, h, w]
                    c_mask = b_c_mask[i]

                    # Compute the error for this trajectory
                    traj_error = get_error(full_frame - results, full_frame, cfg.rollout.error)  # [t, c]
                    valid_c_idx = torch.where(c_mask == 1)[0]
                    traj_error = traj_error[:, valid_c_idx]
                    traj_error = traj_error[gt_ref_steps:, :]  # [t, c]
                    errors_per_type.append(traj_error)

                    if traj_cnt <= cfg.rollout.save and cfg.rollout.save > 0:
                        save_results(
                            results.numpy(force=True),
                            full_frame.numpy(force=True),
                            err=traj_error.mean().item(),
                            folder=f"{cfg.rollout.ckpt_dir.split('/')[-1]}_{cfg.rollout.strategy}",
                            name=f"{dataset_type}_case{traj_cnt}",
                        )

                    if traj_cnt <= 2:  # only print/plot for first 2 trajectories
                        print("")
                        print(dataset_type)
                        print("strategy_list:")
                        print_strategy(strategy_list)
                        print("valid c idx:", valid_c_idx)
                        print("pred traj error shape:", traj_error.shape)

                        if cfg.rollout.save <= 0:
                            plot_traj(
                                f"case{traj_cnt}",
                                dataset_type,
                                results.numpy(force=True),
                                full_frame.numpy(force=True),
                                wandb_run,
                                local=False,
                                upload=cfg.board,
                            )

                if cfg.rollout.save > 0 and traj_cnt >= cfg.rollout.save:
                    break  # stop after saving requested number of trajectories

            else:
                # Original single-trajectory rollout (batch_size=1)
                traj_cnt += 1
                full_frame = b_traj[0]  # [t, c, h, w]
                c_mask = b_c_mask[0]

                # Prepare the pred results
                results = full_frame.new_zeros(full_frame.shape)
                results[:gt_ref_steps, ...] = full_frame[:gt_ref_steps, ...]

                # Rollout
                results = rollout_one_traj(
                    trainer, dataset_type, cfg, results, c_mask, tc_rng, strategy_list
                )  # [t, c, h, w]

                # Compute the mean squared error for the current trajectory
                traj_error = get_error(full_frame - results, full_frame, cfg.rollout.error)  # [t, c]
                valid_c_idx = torch.where(c_mask == 1)[0]
                traj_error = traj_error[:, valid_c_idx]
                traj_error = traj_error[gt_ref_steps:, :]  # [t, c]
                errors_per_type.append(traj_error)

                if traj_cnt <= cfg.rollout.save and cfg.rollout.save > 0:
                    save_results(
                        results.numpy(force=True),
                        full_frame.numpy(force=True),
                        err=traj_error.mean().item(),
                        folder=f"{cfg.rollout.ckpt_dir.split('/')[-1]}_{cfg.rollout.strategy}",
                        name=f"{dataset_type}_case{traj_cnt}",
                    )
                    if traj_cnt == cfg.rollout.save:
                        break

                if traj_cnt <= 2:  # only plot the first 2 trajectories
                    print("")
                    print(dataset_type)
                    print("strategy_list:")
                    print_strategy(strategy_list)
                    print("valid c idx:", valid_c_idx)
                    print("pred traj error shape:", traj_error.shape)

                    if cfg.rollout.save <= 0:
                        plot_traj(
                            f"case{traj_cnt}",
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
