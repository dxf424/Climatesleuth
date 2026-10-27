import climatesleuth
from climatesleuth.datasets import XarrayBalancedBatchDataset
from climatesleuth.models import MeanPooledVGG, NTXentLoss
from climatesleuth.evaluation import evaluate_contrastive_model
from climatesleuth.analysis.backward_optimization import run_backward_optimization
from climatesleuth.analysis.integrated_gradient import compute_integrated_gradients
import xarray as xr
import torch
from torch.utils.data import DataLoader
import argparse
import sys

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run backward optimization and evaluation.")
parser.add_argument("--directory", type=str, required=True,
                help="Path to data and model directory.")
args = parser.parse_args()

# Use the directory from command-line arguments
directory = args.directory

# Assuming you already ran evaluation and have these:
# - model: trained contrastive model (DDP-wrapped or not)
# - X_test: latent features from test set
# - static_input_tensor: torch.Tensor of shape [1, 4, 192, 288]
# - test_dataset: Xarray dataset containing the test data

train_dataset = xr.open_zarr(f"{directory}/data/train_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)
val_dataset   = xr.open_zarr(f"{directory}/data/val_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)
test_dataset  = xr.open_zarr(f"{directory}/data/test_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)

static_ds = xr.open_zarr(f"{directory}/data/static_input.zarr", consolidated=False)
static_input_tensor = torch.tensor(static_ds["static_input"].values, dtype=torch.float32).unsqueeze(0)
static_repeated = static_input_tensor.repeat(2, 1, 1, 1)

model_class = MeanPooledVGG  # or MeanPooledResNet18
save_path = f"{directory}/model/contrastive_model_vgg_seed_42_rank_0_100_accumulation_steps_GELU.pth"

# Example: Extract model and features from evaluate_contrastive_model if it returns them
model, X_test, y_test = evaluate_contrastive_model(
    model_class=model_class,
    save_path=save_path,
    data_structure='weekly',
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    static_repeated=static_repeated,
    batch_size=1,
    num_workers=4,
    in_channels=5,
    projection_dim=128,
    return_model=True,         # <-- Add this keyword if not already supported
    return_feature=True,       # <-- Make sure to get latent space
    for_analysis=True
)

# Make sure model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Run backward optimization
run_backward_optimization(
    model=model,
    X_test=X_test,
    static_input_tensor=static_input_tensor,
    test_dataset=test_dataset,
    device=device,
    num_top_pcs=3,
    num_steps=4000,
    lr=0.01
)

in_channels = 5
projection_dim = 128
batch_size = 1
num_workers = 4
max_batches = 300
n_steps = 10 #at least 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load Datasets
# ------------------------------
print("🔍 Loading datasets...")
batch_test_dataset = XarrayBalancedBatchDataset(test_dataset, shuffle=False)
test_loader = DataLoader(batch_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ------------------------------
# Load Model
# ------------------------------
print("📦 Loading model...")
model = model_class(in_channels=in_channels, projection_dim=projection_dim)
state_dict = torch.load(save_path, map_location=device)
# Remove 'module.' prefix from DDP-trained models if needed
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ------------------------------
# Compute Integrated Gradients
# ------------------------------
print("🧮 Computing Integrated Gradients...")
results = compute_integrated_gradients(
    model=model,
    test_loader=test_loader,
    test_dataset=test_dataset,
    static_input_tensor=static_input_tensor,
    device=device,
    max_batches=max_batches,
    n_steps=n_steps
)
