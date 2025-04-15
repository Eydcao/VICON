import h5py
import numpy as np
import os
from tqdm import tqdm
import glob


def split(path, mode=["train", "valid", "test"], split_percent=[0.8, 0.1, 0.1]):
    # all the h5 files in the path
    # split the files into train, valid, test according to the split_percent
    # mv the files to the corresponding folder with new names (counting start from 0)

    filenames = glob.glob(os.path.join(path, "*.h5"))
    for m in mode:
        filenames += glob.glob(os.path.join(path, m, "*.h5"))
    print(f"Total number of files: {len(filenames)}")
    # shuffle
    np.random.shuffle(filenames)
    # split
    train_size = int(len(filenames) * split_percent[0])
    valid_size = int(len(filenames) * split_percent[1])
    test_size = len(filenames) - train_size - valid_size
    train_files = filenames[:train_size]
    valid_files = filenames[train_size : train_size + valid_size]
    test_files = filenames[train_size + valid_size :]
    # create sub folders for modes
    for m in mode:
        os.makedirs(os.path.join(path, m), exist_ok=True)
    # move files
    # 1. mv to tmp+mode sub folder
    # create tmp+mode sub folders
    for m in mode:
        os.makedirs(os.path.join(path, "tmp" + m), exist_ok=True)
    for im, mode_files in enumerate([train_files, valid_files, test_files]):
        for i, f in enumerate(mode_files):
            # check new file name
            print(f"mv {f} to {os.path.join(path, 'tmp'+mode[im], f'{i}.h5')}")
            os.rename(f, os.path.join(path, "tmp" + mode[im], f"{i}.h5"))
    # 2. rename the folders
    for m in mode:
        os.rename(os.path.join(path, "tmp" + m), os.path.join(path, m))


if __name__ == "__main__":
    path1 = "/data/icon-data/2D/CFD/converted/2D_Train_Rand/"
    split(path1)
    path2 = "/data/icon-data/2D/CFD/converted/2D_Train_Turb/"
    split(path2)
