import torch
from absl import logging
import utils
import torch.optim as optim
import numpy as np
from tabulate import tabulate
import torch.nn as nn
from torchinfo import summary
from crop_utils import NODE_TYPE
from omegaconf import OmegaConf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, model_cfg, opt_cfg, loss_cfg, trainable_mode, amp=False, multi_gpu=False):
        print("flash_sdp_enabled", torch.backends.cuda.flash_sdp_enabled())  # True
        self.model = model
        self.model_cfg = model_cfg
        self.opt_cfg = opt_cfg
        self.trainable_mode = trainable_mode
        self.loss_cfg = loss_cfg
        print("trainable_mode: {}".format(self.trainable_mode), flush=True)
        print("loss_cfg: {}".format(self.loss_cfg), flush=True)

        if multi_gpu and torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            print("model wrapped by DataParallel", flush=True)

        self.device = device
        self.model.to(device)
        print("model moved to {}".format(device), flush=True)

        model = self.model.module if hasattr(self.model, "module") else self.model
        if not (trainable_mode == "all"):  # freeze the model first
            for param in model.parameters():
                param.requires_grad = False

        # Dictionary mapping the component to its parameter name pattern
        patterns = {
            "unet": ["unet"],
            "transformer": ["transformer"],
        }

        for name, params in model.named_parameters():
            for mode, pattern_list in patterns.items():
                if any(pattern in name for pattern in pattern_list) and mode in trainable_mode:
                    params.requires_grad = True

        headers = ["Parameter Name", "Shape", "Requires Grad"]
        table_data = [(name, str(param.shape), param.requires_grad) for name, param in model.named_parameters()]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        headers = ["Trainable Parameters", "Shape"]
        table_data = [(name, str(param.shape)) for name, param in model.named_parameters() if param.requires_grad]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_cfg.peak_lr,
            weight_decay=opt_cfg.weight_decay,
        )
        self.lr_scheduler = utils.WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup=opt_cfg.warmup_steps,
            max_iters=opt_cfg.decay_steps,
        )
        self.amp = amp
        if self.amp:
            self.scaler = torch.amp.GradScaler("cuda")
            print("Using automatic mixed precision", flush=True)

        print(self.model, flush=True)
        print(
            f"Number of parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad]):,}",
            flush=True,
        )
        self.train_step = 0

    def _data_preprocess(self, data):
        if self.loss_cfg.scale == "bc":
            # Check if data is in tuple format (init_data, end_data)
            if isinstance(data, (list, tuple)):
                init_data, end_data, c_mask = data
                # Calculate mean and std using init_data
                mean = torch.mean(init_data, axis=(1, -2, -1), keepdim=True)  # (bs, 1, c_i, 1, 1)
                std = torch.std(init_data, axis=(1, -2, -1), keepdim=True)  # (bs, 1, c_i, 1, 1)
                # NOTE apply channel mask, invalid channel(0) will not be scaled as well
                c_mask = c_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (bs, 1, c_i, 1, 1)
                not_scale = c_mask == 0
                mean = torch.where(not_scale, 0, mean)
                std = torch.where(not_scale, 1, std)
                # NOTE assume the last channel is the type, no need to scale, hence here mean = 0, std = 1
                mean[:, :, -1, :, :] = 0
                std[:, :, -1, :, :] = 1
                if self.model_cfg.preprocess.scheme == "scale_to_one":
                    std = torch.where(std < self.model_cfg.preprocess.eps, 1, std)  # Avoid division by zero
                elif self.model_cfg.preprocess.scheme == "scale_to_eps":
                    std = torch.where(std < self.model_cfg.preprocess.eps, self.model_cfg.preprocess.eps, std)
                else:
                    raise ValueError(
                        "model_cfg['preprocess']['scheme'] {} not supported".format(self.model_cfg.preprocess.scheme)
                    )
                # Normalize both init_data and end_data
                init_data_normalized = (init_data - mean) / std
                end_data_normalized = (end_data - mean) / std  # Use same mean and std for end_data

                return (init_data_normalized, end_data_normalized, c_mask), mean, std

            else:
                # Original logic for data format [bs, pairs, 2, c, h, w]
                mean = torch.mean(data[:, :, 0, :, :, :], axis=(1, -2, -1), keepdim=True)  # (bs, 1, c, 1, 1)
                std = torch.std(data[:, :, 0, :, :, :], axis=(1, -2, -1), keepdim=True)  # (bs, 1, c, 1, 1)
                # NOTE assume the last channel is the type, no need to scale, hence here mean = 0, std = 1
                mean[:, :, -1, :, :] = 0
                std[:, :, -1, :, :] = 1
                data = (data - mean[:, :, None, :, :, :]) / std[:, :, None, :, :, :]  # (bs, pairs, 2, c, h, w)
                return data, mean, std

        elif self.loss_cfg.scale == "none":
            return data, 0, 1

        else:
            raise ValueError("loss_cfg['scale'] {} not supported".format(self.loss_cfg.scale))

    def _model_forward(self, data):
        """
        a wrapper to call model.forward
        data: [bs, pairs, 2, c, h, w]
        """
        output = self.model(data)
        return output

    def _get_label(self, data):
        """
        Extract the label data from the input data.
        """
        if isinstance(data, (list, tuple)):
            # Assuming data is a tuple or list of (init_data, end_data)
            label = data[1]
        else:
            # Assuming data is a single tensor of shape [bs, pairs, 2, c, h, w]
            label = data[:, :, 1]
        return label

    def _loss_fn(self, data):
        """
        Calculate the loss function, taking into account the type channel for masking.
        """
        min_ex = self.loss_cfg.min_ex
        data, _, _ = self._data_preprocess(data)

        label = self._get_label(data)[:, min_ex:]
        c_mask = data[2]  # [bs, 1, c_i, 1, 1]

        # remove the type channel from masks and label
        type_mask = label[:, :, -1:]  # [bs, pairs, 1, h, w]
        c_mask = c_mask[:, :, :-1]  # [bs, 1, c_i - 1, 1, 1]
        label = label[:, :, :-1]

        # Create a mask where type is interior and channel is valid
        valid_mask = (type_mask == NODE_TYPE.INTERIOR.value) * c_mask.float()

        with torch.amp.autocast("cuda", enabled=bool(self.amp), dtype=torch.bfloat16):
            output = self._model_forward(data)[:, min_ex:, :-1]

            # Calculate MSE loss only on the masked region
            mse_loss_fn = nn.MSELoss(reduction="none")
            loss = mse_loss_fn(output, label)
            masked_loss = loss * valid_mask.float()  # Apply mask
            loss = masked_loss.sum() / valid_mask.sum()  # Mean loss over the unmasked elements

        return loss

    def _move_to_device(self, data):
        if isinstance(data, (list, tuple)):
            # If data is a list or tuple, move each item to the device
            return [d.to(self.device) for d in data]
        else:
            # If data is a single tensor
            return data.to(self.device)

    def summary(self, data):
        data = self._move_to_device(data)

        if isinstance(data, (list, tuple)):
            # If data is a list or tuple, summarize each item
            summary(self.model, data[0].size()[1:], data[0].size()[1:])
        else:
            # If data is a single tensor
            summary(self.model, data.size()[1:])

    def iter(self, data):
        """
        train the model, assume data are numpy array
        """
        data = self._move_to_device(data)
        loss = self._loss_fn(data)

        if self.opt_cfg.gradient_accumulation_steps > 1:
            loss = loss / self.opt_cfg.gradient_accumulation_steps

        if not self.amp:  # regular training
            loss.backward()
            if (self.train_step + 1) % self.opt_cfg.gradient_accumulation_steps == 0:
                # Gradient clipping
                model = self.model.module if hasattr(self.model, "module") else self.model
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_cfg.gnorm_clip)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

        else:  # using amp
            self.scaler.scale(loss).backward()
            if (self.train_step + 1) % self.opt_cfg.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                model = self.model.module if hasattr(self.model, "module") else self.model
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_cfg.gnorm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

        self.train_step += 1

    def save(self, save_dir):
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), "{}/{}_params.pth".format(save_dir, self.train_step))
        logging.info("saved to {}, step {}".format(save_dir, self.train_step))

    def restore(self, save_dir, step, restore_opt_state=True):
        params_path = "{}/{}_params.pth".format(save_dir, step)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(torch.load(params_path, map_location=device))
        logging.info("restored params from {}, step {}".format(save_dir, step))
        # TODO: restore optimizer state

    def get_loss(self, data):
        """
        assume raw data
        return numpy loss
        """
        data = self._move_to_device(data)
        loss = self._loss_fn(data)
        return loss.detach().cpu().numpy()

    def get_pred(self, data):
        """
        assume raw data
        return numpy predication
        """
        data = self._move_to_device(data)
        data, mean, std = self._data_preprocess(data)
        with torch.amp.autocast("cuda", enabled=bool(self.amp), dtype=torch.bfloat16):
            output = self._model_forward(data)
        # NOTE skip the last channel (type) of std and mean
        output = output * std + mean
        # NOTE apply channel mask, only valid channel will be kept, others will be set to 0
        c_mask = data[2]  # [bs, 1, c_i, 1, 1]
        # remove the type channel from c_mask
        c_mask = c_mask[:, :, :-1]
        output = output[:, :, :-1] * c_mask.float()
        return output, c_mask

    def get_error(self, types, data, relative=True):
        """
        Calculate the error, excluding the type channel and applying a mask.

        :param types: a vector of string, each is a type of pde
        :param data: Raw data (can be a tensor or tuple/list of tensors).
        :param relative: Boolean to indicate if relative error is to be calculated.

        :return: Mean and standard deviation of the error.
        """
        output_all, c_mask_all = self.get_pred(data)
        output_all = output_all.detach().cpu().numpy()  # (bs, pairs, c, h, w)
        c_mask_all = c_mask_all.detach().cpu().numpy()  # (bs, 1, c, 1, 1)
        label_all = self._get_label(data).detach().cpu().numpy()  # (bs, pairs, c+1(type), h, w)

        # get the set of all types
        type_set = set(types)
        error_mean_dict = {}
        error_std_dict = {}
        # for each type in the set, filter the subset of data depending on where this type is
        # then conduct the stats calculation
        for t in type_set:
            subset_idx = np.array([i for i, x in enumerate(types) if x == t], dtype=int)
            if len(subset_idx) == 0:
                continue

            output = output_all[subset_idx]
            c_mask = c_mask_all[subset_idx]
            label = label_all[subset_idx]

            # Exclude the type channel and get the mask
            label_without_type = label[:, :, :-1]
            type_mask = label[:, :, -1:]  # Assuming the last channel is the mask

            # Apply the mask
            # (0 in type mask means to include the pixel in the error calculation)
            # (1 in channel mask means to include the valid channel in the error calculation)
            mask = (type_mask == 0) * c_mask
            masked_error = np.where(mask, (output - label_without_type) ** 2, 0)

            # Compute the error
            error = np.sqrt(
                np.sum(masked_error, axis=(-2, -1)) / (np.sum(mask, axis=(-2, -1)) + 1e-6)
            )  # (bs, pairs, c)

            if relative:
                # Calculate scale per channel and apply it
                label_sqr = np.where(mask, label_without_type**2, 0)
                label_sqr_mean = np.sum(label_sqr, axis=(0, 1, -2, -1)) / (
                    np.sum(mask, axis=(0, 1, -2, -1)) + 1e-6
                )  # (c,)
                label_scale = np.sqrt(label_sqr_mean) + 1e-6  # (c,)
                error = error / label_scale[None, None, :]  # (bs, pairs, c)

            error_mean = np.mean(error, axis=0)  # (pairs, c)
            error_std = np.std(error, axis=0)  # (pairs, c)

            # build dict
            error_mean_dict[t] = error_mean
            error_std_dict[t] = error_std

        return error_mean_dict, error_std_dict
