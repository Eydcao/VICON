import torch.nn as nn
import torch
from model_utils import patchify, depatchify, build_alternating_block_lowtri_mask, crop_center


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


class ICON_CROPPED(nn.Module):
    def __init__(self, cfg):
        super(ICON_CROPPED, self).__init__()

        self.cfg = cfg

        # read in input cfg and derive some dimensions
        c_in = cfg.transformer.dim_channel
        c_out = cfg.transformer.dim_channel
        dim_token = cfg.transformer.dim_token
        p_i = cfg.patch_num_in
        p_e = cfg.patch_num_out
        p_res = cfg.patch_resolution
        demo_num = cfg.demo_num
        # derive pre/post projection dim
        dim_pre_proj = p_res * p_res * c_in
        dim_post_proj = int(p_res * p_res * c_out * p_e * p_e / p_i / p_i)
        # func_num = 2 * demo_num

        self.pre_proj = nn.Linear(in_features=dim_pre_proj, out_features=dim_token)
        self.post_proj = nn.Linear(in_features=dim_token, out_features=dim_post_proj)

        # the pos_encoding for the end should be cropped from the center of the patch
        patch_pos_encoding_init = torch.randn(p_i, p_i, dim_token)
        if p_e != p_i:
            patch_pos_encoding_end = crop_center(patch_pos_encoding_init, p_e)
        else:
            patch_pos_encoding_end = patch_pos_encoding_init
        # reshape
        self.patch_pos_encoding_init = nn.Parameter(patch_pos_encoding_init.reshape(p_i * p_i, dim_token))
        self.patch_pos_encoding_end = nn.Parameter(patch_pos_encoding_end.reshape(p_e * p_e, dim_token))
        # 2 functional embeddings (n=n_pairs * n_crops) (1 for init, 1 for end)
        self.func_pos_encoding_init = nn.Parameter(torch.randn(demo_num, dim_token))
        self.func_pos_encoding_end = nn.Parameter(torch.randn(demo_num, dim_token))

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_token,
            nhead=cfg.transformer.nhead,
            dim_feedforward=cfg.transformer.dim_feedforward,
            dropout=cfg.transformer.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.transformer.norm_first,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer.num_layers,
            norm=nn.LayerNorm(dim_token) if cfg.transformer.norm_first else None,
        )

        # create the mask with alternative block size (one for the init then another one for the end) for each pair
        mask = (1 - build_alternating_block_lowtri_mask(demo_num, p_i * p_i, p_e * p_e)).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        x: tuple((bs, pairs, c0, h_init, w_init), (bs, pairs, c0, h_end, w_end), c_mask:(bs, c_i))
        bs: batch size
        pairs: effective number of pairs = n_crops * n_pairs
        init: condition; end: QOI
        c0: number of data channels; NOTE input cfg
        h_init/end, w_init/end: height and width of init/end input; NOTE derived from tuple's shape
        h_p, w_p: height and width of each patch, for both init and end, as they should be the same; NOTE input cfg
        p_init/end = self.cfg.patch_num_in/end: number of patches to divide the input image in "each" dimension; NOTE input cfg
        d = self.cfg.transformer.dim_token: dimension of the token; NOTE input cfg
        self.cfg.transformer.patch_size = h_p * w_p * c0, how many scalars in each patch NOTE derived
        """
        # read in input cfg and derive some dimensions
        cfg = self.cfg
        # input channel can contain meta info, such as boundary type
        c_out = cfg.transformer.dim_channel
        dim_token = cfg.transformer.dim_token
        p_i = cfg.patch_num_in
        p_e = cfg.patch_num_out
        p_res = cfg.patch_resolution
        # n_pairs = cfg.n_pairs
        # n_crops = cfg.n_crops
        # effective: means each crop of the pair is used as a pair of condition and QOI; ie an augmentation
        demo_num = cfg.demo_num

        init, end, c_mask = x
        bs, _, _, _, _ = init.shape  # batch number
        feature_init = init.view(-1, *init.shape[-3:])  # (bs * pairs, c, h_init, w_init)
        feature_end = end.view(-1, *end.shape[-3:])  # (bs * pairs, c, h_end, w_end)

        # patchify then cat
        feature_init = patchify(feature_init, patch_num=p_i)  # (bs * pairs, p_i * p_i, c * h * w)
        feature_end = patchify(feature_end, patch_num=p_e)  # (bs * pairs, p_e * p_e, c * h * w)
        feature = torch.cat((feature_init, feature_end), dim=1)  # (bs * pairs, p_i * p_i + p_e * p_e, c * h * w)
        feature = self.pre_proj(feature)  # (bs * pairs, p_i * p_i + p_e * p_e, d_model)

        # apply patch pos encoding for init and end, respectively, if enabled
        if self.cfg.use_patch_pos_encoding:
            # (bs * pairs, p_i * p_i + p_e * p_e, d_model)
            feature[:, : p_i * p_i] = feature[:, : p_i * p_i] + self.patch_pos_encoding_init
            # (bs * pairs, p_i * p_i + p_e * p_e, d_model)
            feature[:, p_i * p_i :] = feature[:, p_i * p_i :] + self.patch_pos_encoding_end
        # (bs, pairs, p_i * p_i + p_e * p_e, d_model)
        feature = feature.view(bs, -1, p_i * p_i + p_e * p_e, dim_token)

        # apply func pos encoding for init and end, respectively, if enabled
        if self.cfg.use_func_pos_encoding:
            # (bs, pairs, p_i * p_i + p_e * p_e, d_model)
            feature[:, :, : p_i * p_i] = feature[:, :, : p_i * p_i] + self.func_pos_encoding_init.view(
                1, -1, 1, dim_token
            )
            # (bs, pairs, p_i * p_i + p_e * p_e, d_model)
            feature[:, :, p_i * p_i :] = feature[:, :, p_i * p_i :] + self.func_pos_encoding_end.view(
                1, -1, 1, dim_token
            )

        # print("Success of applying encodings, the self.cfg.use_patch_pos_encoding and self.cfg.use_func_pos_encoding is", self.cfg.use_patch_pos_encoding, self.cfg.use_func_pos_encoding)
        # exit(1)
        # (bs, pairs * (p_i * p_i + p_e * p_e), d_model)
        feature = feature.view(bs, -1, dim_token)
        # (bs, pairs * (p_i * p_i + p_e * p_e), d_model)
        feature = self.transformer(feature, mask=self.mask)
        # (bs, pairs, p_i * p_i + p_e * p_e, d_model)
        feature = feature.view(bs, demo_num, -1, dim_token)
        # truncate the init part
        # (bs, pairs, p_i * p_i, d_model)
        feature = feature[:, :, : p_i * p_i, :]

        # post project
        # (bs, pairs, p_i * p_i, dim_post_proj = p_res * p_res * c_out * p_e * p_e / p_i / p_i)
        feature = self.post_proj(feature)
        # (bs * pairs, p_i, p_i, dim_post_proj = p_res * p_res * c_out * p_e * p_e / p_i / p_i)
        feature = feature.view(bs * demo_num, p_i, p_i, -1)
        # reshape to p_e * p_e patches
        # (bs * pairs, p_e, p_i//p_e, p_e, p_i//p_e, dim_post_proj = p_res * p_res * c_out * p_e * p_e / p_i / p_i)
        feature = feature.view(bs * demo_num, p_e, p_i // p_e, p_e, p_i // p_e, -1)
        feature = feature.permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(bs * demo_num, p_e * p_e, -1)
        # (bs * pairs, c_out, h_e = p_res * p_num_e, w_e = p_res * p_num_e)
        feature = depatchify(feature, patch_num=p_e, c=c_out, h=p_res, w=p_res)
        # (bs, pairs, c_out, h_e = p_res * p_num_e, w_e = p_res * p_num_e)
        feature = feature.view(bs, demo_num, *feature.shape[-3:])

        return feature
