# NeuralFramework — A Modular Deep Learning Framework for PDE Tasks

NeuralFramework is a research-oriented, plug-and-play framework for learning PDE solution operators and dynamics, targeting tasks such as:

- **One-step prediction** (next-step forecasting)
- **Rollout / autoregressive forecasting** (multi-step temporal prediction)
- **Super-resolution (SR)** and multi-scale reconstruction (planned / WIP)
- Multiple model families: **MLP, UNet, FNO, Transformers, Operator Networks, GNN-like models, Diffusion (planned / WIP)**
- Multiple data modalities: **1D/2D/3D grids, point clouds, graphs** (partially integrated)

The project emphasizes:
- **Registry-based extensibility** (models/datasets are pluggable)
- **Config-first experimentation** (YAML drives everything)
- **DDP-ready training loop**
- **First-class geometry support** (`coords`, `geom`)
- **Dataset preprocessing cache** for fast iteration

---

## Project Structure

```
├── main.py # entrypoint: train (test mode is currently WIP)
├── config.py # CLI args: --mode, --config
├── trainers/
│ └── base.py # BaseTrainer (DDP, logging, ckpt, metrics)
├── forecastors/
│ └── base.py # BaseForecaster (inference/metrics utilities)
├── datasets/
│ ├── base.py # BaseDataset (split/process/cache/loaders)
│ └── ns2d.py # NS2DDataset (Navier–Stokes 2D)
├── models/
│ ├── init.py # MODEL_REGISTRY
│ ├── mlp/ # point-wise MLP
│ ├── unet/ # UNet 1D/2D/3D (layout-specific)
│ ├── fno/ # FNO 1D/2D/3D (layout-specific)
│ ├── transformer/ # Transformer backbone
│ ├── swin_transformer/ # Swin variants
│ ├── m2no/ # multigrid-style operator blocks (layout-specific)
│ ├── ono/, lsm/, gnot/, galerkin_transformer/, transolver/ ...
├── utils/
│ ├── helper.py # seed/device/logger/save_code/save_config
│ ├── loss.py # LossRecord, LpLoss, CompositeLoss
│ ├── metrics.py # Evaluator: rmse/psnr/ssim/...
│ ├── normalizer.py # Gaussian normalizers
│ └── rollout.py # autoregressive rollout util (generic)
└── template/config/
├── base.yaml
└── ns2d.yaml # NOTE: templates may be outdated; see config examples below
```

---

## Installation

This repo does not ship a pinned `requirements.txt` yet. Typical dependencies:

- Python >= 3.9
- PyTorch
- numpy, scipy, h5py, pyyaml, tqdm
- wandb (optional)

Example:

```bash
pip install torch numpy scipy h5py pyyaml tqdm wandb
```