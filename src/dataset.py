import torch
import functools
from typing import Optional
from torch.utils.data import Dataset, IterableDataset, DataLoader
import os
import glob
import h5py
import numpy as np
import torch.nn.functional as F
import torchdata.datapipes as dp
import math
from dataset_utils import crop_split_data, split_data
from crop_utils import NODE_TYPE

DatasetIdx = {"NS2D": 0, "COMPRESSIBLE2D": 1, "EULER2D": 2}

# NOTE: below hard code the channel physical meanings, do not remove this comment
# [rho, vx, vy, pre, vor, scalar, int_bc_type]


def _augment_data(cur_data, rng, tc_rng, cfg, split_scheme, pad_mode):
    if split_scheme == "raw":
        return cur_data
    elif split_scheme == "crop_split":
        pairs, t_in, t_out, delta_t = crop_split_data(cur_data, tc_rng, cfg, pad_mode)
        return cur_data, pairs, t_in, t_out, delta_t
    elif split_scheme == "split":
        pairs, t_in, t_out, delta_t = split_data(cur_data, rng, tc_rng, cfg)
        return cur_data, pairs, t_in, t_out, delta_t
    else:
        raise ValueError("split_scheme {} not supported".format(split_scheme))


def _train_filter(fname):
    return "train" in fname and "h5" in fname


def _valid_filter(fname):
    return "valid" in fname and "h5" in fname


def _test_filter(fname):
    return "test" in fname and "h5" in fname


def _get_all_in_one_path(type_cfg, mode):
    # NOTE assume the data h5s are all in one folder, but with different mode in the file name
    saved_folder = type_cfg.folder
    if mode == "train":
        file_list = list(
            map(
                lambda filename: os.path.join(saved_folder, filename),
                filter(_train_filter, os.listdir(saved_folder)),
            )
        )
    elif mode == "valid":
        file_list = list(
            map(
                lambda filename: os.path.join(saved_folder, filename),
                filter(_valid_filter, os.listdir(saved_folder)),
            )
        )
    elif mode == "test":
        file_list = list(
            map(
                lambda filename: os.path.join(saved_folder, filename),
                filter(_test_filter, os.listdir(saved_folder)),
            )
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return list(sorted(file_list))


def _get_nested_path(type_cfg, mode):
    # NOTE the path is path/mode
    saved_folder = type_cfg.folder
    if mode == "train" or mode == "valid" or mode == "test":
        final_folder = os.path.join(saved_folder, mode)
        # find all files end with .h5
        file_list = glob.glob(os.path.join(final_folder, "*.h5"))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return list(sorted(file_list))


class datasetBase(dp.iter.IterDataPipe):
    """
    Base class for all iterable datasets, and contains some shared helper methods.
    Added support for dropped frames during rollout.
    """

    def _hash(self, list, base):
        # from left to right list[0]*base^0 + list[1]*base^1 + ... + list[n]*base^n
        hash = 0
        for i in range(len(list)):
            hash += list[i] * (base**i)
        return hash

    def _init_rng(self):
        """
        Initialize different random generator for each worker.
        """
        if self.rng is not None:
            return

        worker_id, _ = self._get_worker_id_and_info()
        self.worker_id = worker_id
        cfg = self.cfg
        if self.train:
            train_seed = np.random.randint(1_000_000_000)
            seed_list = [train_seed, DatasetIdx[self.data_type], worker_id]
            seed = self._hash(seed_list, 1000)
            self.rng = np.random.default_rng(seed)
            self.tc_rng = torch.Generator()
            self.tc_rng.manual_seed(seed)
        else:
            seed_list = [self.base_seed, DatasetIdx[self.data_type], worker_id]
            seed = self._hash(seed_list, 1000)
            self.rng = np.random.default_rng(seed)
            self.tc_rng = torch.Generator()
            self.tc_rng.manual_seed(seed)

    def _get_worker_id_and_info(self):
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id, worker_info

    def _get_nested_paths(self):
        # split data paths based number of workers
        if self.num_workers <= 1:
            return [self.dp]
        else:
            return np.array_split(self.dp, self.num_workers)

    def _get_dataset_dp(self, data_cfg, mode):
        # inform the user to implement this method in children class
        print("Please implement this method: _get_dataset_dp in children class")
        raise NotImplementedError

    def _load_dataset_file(self, path, mode):
        # inform the user to implement this method in children class
        print("Please implement this method: _load_dataset_file in children class")
        raise NotImplementedError

    def _proc_data(self, data, idx):
        # inform the user to implement this method in children class
        print("Please implement this method: _proc_data in children class")
        raise NotImplementedError

    def __init__(self, cfg, num_workers, base_seed=1, mode="train", rollout_cfg=None):
        # NOTE: must set data_type in children's __init__ then call super().__init__ to init the base class
        self.cfg = cfg
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.epoch = 0
        self.train = mode == "train"
        self.rollout = mode == "rollout"
        self.limit_trajectories = cfg.num_samples_max
        self.rng = None
        self.tc_rng = None

        # Check if we're using dropped frames in rollout mode
        self.dropped_frames = rollout_cfg.dropped

        split_scheme = cfg.split_scheme
        if mode == "rollout":
            # NOTE we do not have file named with rollout, we are using 'test' mode for later file filtering
            mode = "test"
        self.mode = mode
        self.split_scheme = split_scheme
        # get dp
        data_cfg = cfg.types[self.data_type]
        self.dp = self._get_dataset_dp(data_cfg, mode)

    def __iter__(self):
        self._init_rng()
        # get the sub path list for each worker
        worker_id, worker_info = self._get_worker_id_and_info()
        worker_paths = self._get_nested_paths()[worker_id]
        if (self.mode != "test") and self.rng is not None:
            self.rng.shuffle(worker_paths)

        for path in worker_paths:
            try:
                f, data, num_trajs = self._load_dataset_file(path, self.mode)
            except KeyError:
                raise ValueError(f"Data type {self.data_type} not registered")
            except:
                print(f"sth wrong with file {self.data_type} and {path}")

            # get chunk size limit for each worker
            if self.limit_trajectories is not None and self.limit_trajectories != -1:
                # 2. get chunk size limit for each worker
                chunk_size = min(self.limit_trajectories, num_trajs)
            else:
                # no limit, load all
                chunk_size = num_trajs
            # select the chunk of data for each worker
            traj_ids = np.arange(num_trajs, dtype=int)
            if (self.mode != "test") and self.rng is not None:
                self.rng.shuffle(traj_ids)
            worker_chunk_ids = traj_ids[:chunk_size]

            for idx in worker_chunk_ids:
                # 3.5 get c_mask
                c_mask = torch.tensor(np.array(self.cfg.types[self.data_type].c_mask), dtype=torch.float32)

                # 3. proc data before augmentation (such as cropping, rotation and flipping), case by case
                if self.rollout and self.dropped_frames:
                    cur_data, dropped_mask = self._proc_data(data, idx)
                else:
                    cur_data = self._proc_data(data, idx)

                # 4. augmentation of data, common
                if not self.rollout:
                    cur_data, pairs, t_in, t_out, delta_t = _augment_data(
                        cur_data,
                        self.rng,
                        self.tc_rng,
                        self.cfg,
                        self.split_scheme,
                        self.cfg.types[self.data_type].pad_mode,
                    )
                    # append the c_mask to pairs so can be used in model later
                    pairs = list(pairs)
                    pairs.append(c_mask)
                    # NOTE cannot yield cur_data due to different shape
                    yield self.data_type, pairs, t_in, t_out, delta_t
                else:
                    if self.dropped_frames:
                        yield (self.data_type, cur_data, dropped_mask, c_mask)
                    else:
                        yield (self.data_type, cur_data, c_mask)


class NS2DDataset(datasetBase):
    """
    PDEArena-Incomp: Incompressible Navier-Stokes Dataset.

    PDE: ∂_t u + u·∇u = -∇p + μΔu + f,  ∇·u = 0
    Domain: [0, 32]² × [18, 105], dt = 1.5, resolution 128 × 128
    Trajectories: 2,496/608/608 (train/valid/test), 56 timesteps each
    BCs: Velocity (Dirichlet), scalar particle field (Neumann)
    Features: density (zero), vx, vy, pressure (zero), vorticity (zero), scalar (u), node_type

    Dataset structure:
        HDF5 file with groups 'train', 'valid', 'test', each containing:
        - u: [N, T, H, W] scalar particle field
        - vx: [N, T, H, W] x-velocity
        - vy: [N, T, H, W] y-velocity

    Reference: Gupta & Brandstetter (2022), PDEArena
    """

    def __init__(self, cfg, num_workers, base_seed, mode="train", rollout_cfg=None):
        self.data_type = "NS2D"
        super().__init__(cfg, num_workers, base_seed, mode, rollout_cfg)

    def _get_dataset_dp(self, data_cfg, mode):
        return _get_all_in_one_path(data_cfg, mode)

    def _load_dataset_file(self, path, mode):
        f = h5py.File(path, "r")
        data = f[mode]
        num = data["u"].shape[0]

        return f, data, num

    def _proc_data(self, data, idx):
        u = torch.tensor(data["u"][idx], dtype=torch.float32).unsqueeze(1)
        zero = torch.zeros_like(u, dtype=torch.float32)

        node_type = torch.ones_like(u) * NODE_TYPE.INTERIOR.value  # [t, nx, ny]
        # set the boundary nodes (nx, and ny = 0 or = -1) to be boundary
        node_type[:, :, 0] = NODE_TYPE.BOUNDARY.value
        node_type[:, :, -1] = NODE_TYPE.BOUNDARY.value
        node_type[:, :, :, 0] = NODE_TYPE.BOUNDARY.value
        node_type[:, :, :, -1] = NODE_TYPE.BOUNDARY.value

        vx = torch.tensor(data["vx"][idx], dtype=torch.float32)
        vy = torch.tensor(data["vy"][idx], dtype=torch.float32)

        vel = torch.cat((vx[:, None], vy[:, None]), dim=1)

        # [zero, vel, zero, zero, u, type]
        processed_data = torch.cat([zero, vel, zero, zero, u, node_type], dim=1)
        processed_data = processed_data.contiguous()

        # Check if we're in rollout mode with dropped frames and dropped_mask exists
        if self.rollout and self.dropped_frames:
            dropped_mask = torch.tensor(data["dropped_mask"][idx], dtype=torch.bool)
            return processed_data, dropped_mask

        return processed_data


class compressible2DDatasetBase(datasetBase):
    """
    PDEBench Compressible Navier-Stokes Base Dataset.

    PDE (Compressible NS):
        ∂_t ρ + ∇·(ρu) = 0
        ρ(∂_t u + u·∇u) = -∇p + η∆u + (ζ + η/3)∇(∇·u)
        ∂_t(ε + ρu²/2) = -∇·[(ε + p + ρu²/2)u - u·σ']
    Domain: T² × [0, 1], dt = 0.05, resolution 128 × 128
    BCs: Periodic (torus)
    Features: density, vx, vy, pressure, vorticity (zero), scalar (zero), node_type

    Dataset structure:
        HDF5 files (one per trajectory) containing:
        - Vx: [T, H, W] x-velocity
        - Vy: [T, H, W] y-velocity
        - pressure: [T, H, W] pressure field
        - density: [T, H, W] density field

    Reference: Takamoto et al. (2022), PDEBench
    """

    def __init__(self, cfg, num_workers, base_seed, mode="train", rollout_cfg=None, inviscid=False):
        self.data_type = "COMPRESSIBLE2D" if not inviscid else "EULER2D"
        super().__init__(cfg, num_workers, base_seed, mode, rollout_cfg)

    def _get_dataset_dp(self, data_cfg, mode):
        return _get_nested_path(data_cfg, mode)

    def _load_dataset_file(self, path, mode):
        f = h5py.File(path, "r")
        data = f

        num = data["pressure"].shape[0]

        return f, data, num

    def _proc_data(self, data, idx):
        u = torch.tensor(data["Vx"][idx], dtype=torch.float32)  # [t, nx, ny]
        v = torch.tensor(data["Vy"][idx], dtype=torch.float32)
        node_type = torch.ones_like(u) * NODE_TYPE.INTERIOR.value  # [t, nx, ny]
        # do we need to change node type for compressible flow? NOTE no need

        pres = torch.tensor(data["pressure"][idx], dtype=torch.float32)  # [t, nx, ny]
        density = torch.tensor(data["density"][idx], dtype=torch.float32)  # [t, nx, ny]

        zero = torch.zeros_like(pres)

        # [u,v,pres,density,zero,type]
        processed_data = torch.stack([density, u, v, pres, zero, zero, node_type], dim=1)
        processed_data = processed_data.contiguous()

        # Check if we're in rollout mode with dropped frames and dropped_mask exists
        if self.rollout and self.dropped_frames:
            dropped_mask = torch.tensor(data["dropped_mask"][idx], dtype=torch.bool)
            return processed_data, dropped_mask

        return processed_data


class compressible2DDataset(compressible2DDatasetBase):
    """
    PDEBench-Comp-HighVis: Compressible Navier-Stokes with High Viscosity.

    Inherits from compressible2DDatasetBase.
    Viscosity: Higher values (various combinations of shear and bulk viscosity)
    Trajectories: 40,000 total, 21 timesteps each
    Split: 80%/10%/10% (train/valid/test)

    See compressible2DDatasetBase for PDE details and data structure.
    """

    def __init__(self, cfg, num_workers, base_seed, mode="train", rollout_cfg=None):
        super().__init__(cfg, num_workers, base_seed, mode, rollout_cfg, False)


class Euler2DDataset(compressible2DDatasetBase):
    """
    PDEBench-Comp-LowVis: Compressible Navier-Stokes with Low Viscosity (Euler).

    Inherits from compressible2DDatasetBase.
    Viscosity: Numerically zero (1e-8), essentially inviscid Euler equations
    Resolution: Raw 512 × 512, downsampled to 128 × 128 via average pooling
    Trajectories: 4,000 total, 21 timesteps each
    Split: 80%/10%/10% (train/valid/test)

    See compressible2DDatasetBase for PDE details and data structure.
    """

    def __init__(self, cfg, num_workers, base_seed, mode="train", rollout_cfg=None):
        super().__init__(cfg, num_workers, base_seed, mode, rollout_cfg, True)


DatasetHandler = {
    "NS2D": NS2DDataset,
    "COMPRESSIBLE2D": compressible2DDataset,
    "EULER2D": Euler2DDataset,
}


def all_datasets(cfg, num_workers, base_seed, mode, rollout_cfg):
    datasets = {}
    for type in cfg.types.keys():
        datasets[type] = DatasetHandler[type](cfg, num_workers, base_seed, mode, rollout_cfg)
    return datasets
