import h5py
import numpy as np
import os
from tqdm import tqdm


def convert(folders, save_folder, split_size):
    os.makedirs(save_folder, exist_ok=True)
    paths = []
    for folder in folders:
        filenames = os.listdir(folder)
        paths += sorted([os.path.join(folder, filename) for filename in filenames if "512" in filename])

    save_file_idx = 0

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
                        data_dict[key] = np.mean(np.array(f[key]).reshape(-1, 4), axis=1)
                    else:
                        data = np.array(f[key][cur_line : cur_line + split_size])  # (size, t, nx, ny)
                        data = data.reshape(
                            data.shape[0], data.shape[1], data.shape[2] // 4, 4, data.shape[3] // 4, 4
                        ).mean(
                            axis=(-3, -1)
                        )  # (size, t, nx/4, ny/4)
                        data_dict[key] = data

                traj_path = os.path.join(save_folder, f"{save_file_idx}.h5")
                save_file_idx += 1
                cur_line += split_size
                with h5py.File(traj_path, "w") as g:
                    for key in data_dict.keys():
                        g.create_dataset(key, data=data_dict[key])


if __name__ == "__main__":
    folders = ["/data/icon-data/2D/CFD/2D_Train_Rand/", "/data/icon-data/2D/CFD/2D_Train_Turb/"]
    save_folder = "/data/icon-data/2D/CFD/converted/2D_Train_Turb/"

    convert(folders, save_folder, 50)
