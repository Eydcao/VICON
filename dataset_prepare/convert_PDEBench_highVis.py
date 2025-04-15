import h5py
import numpy as np
import os
from tqdm import tqdm


def convert(folder, save_folder, split_size):
    os.makedirs(save_folder, exist_ok=True)

    filenames = os.listdir(folder)

    save_file_idx = 0

    # for pattern in [128, 512]:
    for pattern in [128]:
        paths = sorted([os.path.join(folder, filename) for filename in filenames if str(pattern) in filename])
        step = pattern // 128

        for path in paths:
            with h5py.File(path, "r") as f:
                total_size = f["density"].shape[0]
                cur_line = 0
                assert total_size % split_size == 0
                for _ in tqdm(range(total_size // split_size)):
                    data_dict = {}
                    for key in f.keys():
                        if key == "t-coordinate":
                            data_dict[key] = np.array(f[key])
                        elif key.endswith("coordinate"):
                            data_dict[key] = np.array(f[key][::step])
                        else:
                            data_dict[key] = np.array(f[key][cur_line : cur_line + split_size, :, ::step, ::step])

                    traj_path = os.path.join(save_folder, f"{save_file_idx}.h5")
                    save_file_idx += 1
                    cur_line += split_size
                    with h5py.File(traj_path, "w") as g:
                        for key in data_dict.keys():
                            g.create_dataset(key, data=data_dict[key])


if __name__ == "__main__":
    folder = "/data/icon-data/2D/CFD/2D_Train_Rand/"
    save_folder = "/data/icon-data/2D/CFD/converted/2D_Train_Rand/"

    convert(folder, save_folder, 500)
