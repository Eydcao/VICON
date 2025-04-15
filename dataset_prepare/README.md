## Dataset preprocessing

### 1. PDEBench Datasets

We split two distinct systems from PDEBench's 2D CFD dataset:

- Compressible flow with high viscosity: PDEBench-Comp-HighVis
- Compressible flow with low viscosity: PDEBench-Comp-LowVis

To download:

```bash
# Clone PDEBench repository
git clone https://github.com/pdebench/PDEBench.git
cd PDEBench
# Download 2D CFD dataset
python pdebench/data_download/download_direct.py --root_folder $YOUR_DATA_PATH --pde_name 2d_cfd
```

To process the PDEBench data, navigate to the `$THIS_REPO/dataset_prepare` folder and run the following scripts in order:

```bash
# Convert high viscosity dataset
python convert_PDEBench_highVis.py
# Convert low viscosity dataset
python convert_PDEBench_lowVis.py
# Split datasets into train/val/test
python split_PDEBench.py
```

You will need to modify the folder locations in these scripts.

### 2. PDEArena Dataset

We use the Navier-Stokes-2D conditioning dataset: PDEArena-Incomp

```bash
# Install git-lfs (required for efficient download)
sudo apt-get install git-lfs
git lfs install
# Download dataset
git clone https://huggingface.co/datasets/pdearena/NavierStokes-2D-conditoned
```

Note: If git-lfs installation is not possible due to permission restrictions, alternative download methods are available on the [PDEArena website](https://microsoft.github.io/PDEArena/download).

**You will need to remove one corrupted file here**：

```bash
rm /data/icon-data/pdeareana/NavierStokes-2D-conditoned/NavierStokes2D_train_496019_0.38774_32.h5
```
