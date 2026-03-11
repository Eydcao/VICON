import torch.nn as nn
import torch
from model_utils import patchify, depatchify, build_alternating_block_lowtri_mask


class ICON_UNCROPPED(nn.Module):
    def __init__(self, cfg):
        super(ICON_UNCROPPED, self).__init__()

        self.cfg = cfg
        assert cfg["patch_num_in"] == cfg["patch_num_out"]

        self.pre_proj = nn.Linear(
            in_features=cfg["transformer"]["dim_channel"] * cfg["patch_resolution"] ** 2,
            out_features=cfg["transformer"]["dim_token"],
        )
        self.post_proj = nn.Linear(
            in_features=cfg["transformer"]["dim_token"],
            out_features=cfg["transformer"]["dim_channel"] * cfg["patch_resolution"] ** 2,
        )

        self.patch_pos_encoding = nn.Parameter(
            torch.randn(cfg["patch_num_in"] * cfg["patch_num_in"], cfg["transformer"]["dim_token"])
        )
        self.func_pos_encoding = nn.Parameter(torch.randn(cfg["demo_num"] * 2, cfg["transformer"]["dim_token"]))

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["transformer"]["dim_token"],
            nhead=cfg["transformer"]["nhead"],
            dim_feedforward=cfg["transformer"]["dim_feedforward"],
            dropout=cfg["transformer"]["dropout"],
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg["transformer"]["num_layers"])

        mask = (
            1
            - build_alternating_block_lowtri_mask(
                cfg["demo_num"], cfg["patch_num_in"] * cfg["patch_num_in"], cfg["patch_num_out"] * cfg["patch_num_out"]
            )
        ).bool()
        self.register_buffer("mask", mask)

    def forward(self, x_tuple):
        """
        x: tuple((bs, pairs, c0, h_init, w_init), (bs, pairs, c0, h_end, w_end), c_mask:(bs, c_i))
        pairs is no larger then cfg.demo_num
        """
        init, end, c_mask = x_tuple
        x = torch.cat((init[:, :, None, :, :, :], end[:, :, None, :, :, :]), dim=2)  # (bs, pairs, 2, c, h, w)
        p = self.cfg["patch_num_in"]
        d = self.cfg["transformer"]["dim_token"]
        bs, pairs, _, c, h, w = x.shape  # _ = 2
        feature = x.view(-1, *x.shape[-3:])  # (bs * pairs * 2, c, h, w)

        c, ph, pw = feature.shape[-3:]
        h = ph // p
        w = pw // p
        feature = patchify(feature, patch_num=p)  # (bs * pairs * 2, p * p, c * h * w)

        feature = self.pre_proj(feature)  # (bs * pairs * 2, p * p, d_model)

        if self.cfg.use_patch_pos_encoding:
            feature = feature + self.patch_pos_encoding  # (bs * pairs * 2, p * p, d_model)
        feature = feature.view(bs, -1, p * p, d)  # (bs, pairs * 2, p * p, d_model)

        if self.cfg.use_func_pos_encoding:
            func_pos_encoding = self.func_pos_encoding.view(1, -1, 1, d)  # (1, cfg["demo_num"] * 2, 1, d_model)
            func_pos_encoding = func_pos_encoding[:, : pairs * 2, :, :]  # (1, pairs * 2, 1, d_model)
            feature = feature + func_pos_encoding  # (bs, pairs * 2, p * p, d_model)

        # print("Success of applying encodings, the self.patch_pos_encoding and self.func_pos_encoding is", self.patch_pos_encoding, self.func_pos_encoding)
        # exit(1)

        feature = feature.view(bs, -1, d)  # (bs, pairs * 2 * p * p, d_model)

        mask = self.mask[: pairs * 2 * p * p, : pairs * 2 * p * p]  # (pairs * 2 * p * p, pairs * 2 * p * p)
        feature = self.transformer(feature, mask=mask)  # (bs, pairs * 2 * p * p, d_model)
        feature = feature.view(bs, pairs, 2, p * p, d)  # (bs, pairs, 2, p * p, d_model)
        feature = feature[:, :, 0, :, :]  # (bs, pairs, p * p, d_model)

        feature = self.post_proj(feature)  # (bs, pairs, p * p, c * h * w)

        feature = feature.view(bs * pairs, *feature.shape[-2:])  # (bs * pairs, p * p, c * h * w)
        feature = depatchify(feature, patch_num=p, c=c, h=h, w=w)  # (bs * pairs, c, ph, pw)
        feature = feature.view(bs, pairs, *feature.shape[-3:])  # (bs, pairs, c, ph, pw)

        return feature
