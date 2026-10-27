import argparse
import climatesleuth
import xarray as xr
import torch
from climatesleuth.analysis.backward_optimization import run_backward_optimization
from climatesleuth.models import MeanPooledVGG, NTXentLoss
from climatesleuth.evaluation import evaluate_contrastive_model
import argparse

# Assuming you already ran evaluation and have these:
# - model: trained contrastive model (DDP-wrapped or not)
# - X_test: latent features from test set
# - static_input_tensor: torch.Tensor of shape [1, 4, 192, 288]
# - test_dataset: Xarray dataset containing the test data

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run backward optimization and evaluation.")
parser.add_argument("--directory", type=str, required=True,
                    help="Path to data and model directory.")
args = parser.parse_args()

# Use the directory from command-line arguments
directory = args.directory
#directory = "/glade/work/fanda/miles-climatesleuth/microphysics_analysis"

train_dataset = xr.open_zarr(f"{directory}/data/train_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)
val_dataset   = xr.open_zarr(f"{directory}/data/val_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)
test_dataset  = xr.open_zarr(f"{directory}/data/test_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)

static_ds = xr.open_zarr(f"{directory}/data/static_input.zarr", consolidated=False)
static_input_tensor = torch.tensor(static_ds["static_input"].values, dtype=torch.float32).unsqueeze(0)
static_repeated = static_input_tensor.repeat(2, 1, 1, 1)

model_class = MeanPooledVGG  # or MeanPooledResNet18

# Example: Extract model and features from evaluate_contrastive_model if it returns them
model, X_test, y_test = evaluate_contrastive_model(
    model_class=model_class,
    save_path=f"{directory}/model/contrastive_model_vgg_seed_42_rank_0_100_accumulation_steps_GELU.pth",
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
