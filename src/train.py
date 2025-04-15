import wandb
import random
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from pprint import pprint
import numpy as np
import pytz
from datetime import datetime
import utils

from dataset import all_datasets
from dataset_utils import InfiniteDataLooper
import models
from trainer import Trainer
from train_utils import board_loss, print_error, eval_plot, get_data_from_looper


def run_train(cfg):
    utils.set_seed(cfg.train_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.train_seed)

    print(OmegaConf.to_yaml(cfg))

    if cfg.board:
        wandb.init(
            project=cfg.wandb.project,
            entity="liuyang-research",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Creation of instances
    # create model
    if cfg.model.type == "crop":
        model = models.ICON_CROPPED(cfg.model)
        print("Using cropped (new) model")
    elif cfg.model.type == "nocrop":
        model = models.ICON_UNCROPPED(cfg.model)
        print("Using uncropped (ancient) model")
    else:
        raise ValueError("Unknown model type: {}".format(cfg.model.type))

    # create dataset
    train_datasets = all_datasets(cfg.datasets, cfg.dataset_workers, cfg.train_seed, "train")
    test_datasets = all_datasets(cfg.datasets, cfg.dataset_workers, cfg.test_seed, "test")

    # create trainer
    trainer = Trainer(
        model, cfg.model, cfg.opt, cfg.loss, trainable_mode=cfg.trainable_mode, amp=cfg.amp, multi_gpu=cfg.multi_gpu
    )
    # create data loaders
    train_loaders = {
        k: torch.utils.data.DataLoader(
            v,
            batch_size=cfg.datasets.types[k].train_batch_size,
            num_workers=cfg.dataset_workers,
            pin_memory=True,
        )
        for k, v in train_datasets.items()
    }
    test_loaders = {
        k: torch.utils.data.DataLoader(
            v,
            batch_size=cfg.datasets.types[k].test_batch_size,
            # num_workers=cfg.dataset_workers,
            num_workers=1,
            pin_memory=True,
        )
        for k, v in test_datasets.items()
    }

    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    stamp = time_stamp
    print("stamp: {}".format(stamp))

    # Training
    train_loopers = {k: InfiniteDataLooper(v) for k, v in train_loaders.items()}
    test_loopers = {k: InfiniteDataLooper(v) for k, v in test_loaders.items()}

    total_steps = cfg.epochs * cfg.steps_per_epoch
    for step in range(total_steps + 1):

        (train_type_list, train_pairs_list, _, _, train_types, train_pairs, _, _) = get_data_from_looper(
            train_loopers, tc_rng, cfg
        )

        trainer.model.eval()

        # log loss
        if (
            (trainer.train_step % cfg.loss_freq == 0)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step <= cfg.loss_freq)
            or (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.loss_freq)
        ):
            # train loss and error
            with torch.inference_mode():
                # log train loss for all datasets
                print_error(train_types, trainer, train_pairs, "train")
                board_loss(trainer, train_pairs, "train", cfg)
                for i in range(len(train_type_list)):
                    # log train loss for each dataset
                    board_loss(trainer, train_pairs_list[i], f"train_{train_type_list[i][0]}", cfg)

                # test loss and error
                (test_type_list, test_pairs_list, _, _, test_types, test_pairs, _, _) = get_data_from_looper(
                    test_loopers, tc_rng, cfg
                )
                # log test loss for all datasets
                print_error(test_types, trainer, test_pairs, "test")
                board_loss(trainer, test_pairs, "test", cfg)
                for i in range(len(test_type_list)):
                    # log test loss for each dataset
                    board_loss(trainer, test_pairs_list[i], f"test_{test_type_list[i][0]}", cfg)

        # log plot
        if cfg.plot and (
            (trainer.train_step % cfg.plot_freq == 0)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step <= cfg.plot_freq)
            or (trainer.train_step % (cfg.plot_freq // 10) == 0 and trainer.train_step >= total_steps - cfg.plot_freq)
        ):
            # NOTE since node bs represent different pde types, we can only plot more than 1 type TODO
            for tmp_type, looper in test_loopers.items():
                tmp_types, tmp_pairs, tmp_in_idx, tmp_out_idx, _ = next(looper)
                eval_plot(tmp_types, trainer, tmp_pairs, f"test_{tmp_type}", tmp_in_idx, tmp_out_idx, 0, cfg)

        if cfg.board and trainer.train_step % (cfg.save_freq) == 0:
            ckpt_dir = f"{cfg.dump_dir}/ckpts/{cfg.project}/" + stamp
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            print("current time: " + datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S"))
            trainer.save(ckpt_dir)

        trainer.model.train()

        trainer.iter(train_pairs)

        if trainer.train_step == cfg.time_warm:  # exclude warming up steps
            utils.timer.tic("time estimate")
        if (trainer.train_step - cfg.time_warm) > 0 and ((trainer.train_step - cfg.time_warm) % cfg.time_freq == 0):
            ratio = (trainer.train_step - cfg.time_warm) / (cfg.epochs * cfg.steps_per_epoch)
            utils.timer.estimate_time("time estimate", ratio)

    if cfg.board:
        wandb.finish()


@hydra.main(version_base=None, config_path="../configs/", config_name="default")
def main(cfg: DictConfig):
    run_train(cfg)


if __name__ == "__main__":
    main()
