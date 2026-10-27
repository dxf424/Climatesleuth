import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import xarray as xr
from mpi4py import MPI
import climatesleuth
from climatesleuth.models import MeanPooledVGG, NTXentLoss
from climatesleuth.datasets import XarrayBalancedBatchPrefetcher, BatchForecastLenDataLoader
from climatesleuth.trainer import ContrastiveTrainer
from typing import Any, Callable, Dict, List, Optional, Tuple
import dask
from concurrent.futures import ProcessPoolExecutor
import random
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run backward optimization and evaluation.")
parser.add_argument("--directory", type=str, required=True,
                    help="Path to data and model directory.")
args = parser.parse_args()

# Use the directory from command-line arguments
directory = args.directory

# Enable dask for parallel computation
dask.config.set(scheduler='threads')  # Uses multi-threading
def seed_everything(seed=1234):
    """Seed all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use for all random number generators. Default is 1234.

    Notes:
        This function seeds:
        - Python's random module
        - NumPy's random module
        - PyTorch's CPU and GPU random number generators
        - Sets environment variable to control hash seed
        - Configures PyTorch's cuDNN to be deterministic and benchmark mode.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

seed = 1
seed_everything(seed)

# DDP config from environment variables
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

# System tuning
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# DDP setup
def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size, seed, directory):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 💡 Model and DDP wrap
    model = MeanPooledVGG(in_channels=5, projection_dim=128).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Loss, Optimizer, AMP
    criterion = NTXentLoss(temperature=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()

    # 📦 Dataset loading
    print(f"[Rank {rank}] Loading datasets...")

    train_ds = xr.open_zarr(f"{directory}/data/train_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)
    val_ds   = xr.open_zarr(f"{directory}/data/val_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)
    static_ds = xr.open_zarr(f"{directory}/data/static_input.zarr", consolidated=False)
    static_input_tensor = torch.tensor(static_ds["static_input"].values, dtype=torch.float32).unsqueeze(0)
    static_repeated = static_input_tensor.repeat(2, 1, 1, 1)  # Add batch dim (2 for pairs)

    # 🚀 Dataset + Prefetching
    train_dataset = XarrayBalancedBatchPrefetcher(
        xarray_ds=train_ds,
        static_repeated=static_repeated,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        rank=rank,
        world_size=world_size,
    )

    val_dataset = XarrayBalancedBatchPrefetcher(
        xarray_ds=val_ds,
        static_repeated=static_repeated,
        shuffle=False,
        num_workers=4,
        prefetch_factor=4,
        rank=rank,
        world_size=world_size,
    )

    train_loader = BatchForecastLenDataLoader(train_dataset)
    val_loader = BatchForecastLenDataLoader(val_dataset)

    print(f"[Rank {rank}] Data ready.")

    # 🎯 Launch Trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        rank=rank,
        world_size=world_size,
        seed=seed,
        accumulation_steps=100,
        patience=10,
        epochs=100,
        save_prefix="contrastive_model_vgg_test"
    )

    trainer.run()


# 🚀 Entry point
if __name__ == "__main__":
    rank = LOCAL_RANK
    world_size = WORLD_SIZE
    seed = 42  # Reproducibility
    main(rank, world_size, seed, directory)
