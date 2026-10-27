import torch
import xarray as xr
from torch.utils.data import DataLoader
import climatesleuth
from climatesleuth.models import MeanPooledVGG, MeanPooledResNet18
from climatesleuth.datasets import XarrayBalancedBatchDataset
from climatesleuth.analysis.integrated_gradient import compute_integrated_gradients
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run backward optimization and evaluation.")
parser.add_argument("--directory", type=str, required=True,
                    help="Path to data and model directory.")
args = parser.parse_args()

# Use the directory from command-line arguments
directory = args.directory

# ------------------------------
# Configuration
# ------------------------------
model_class = MeanPooledVGG  # or MeanPooledResNet18
save_path = f"{directory}/model/contrastive_model_vgg_seed_42_rank_0_100_accumulation_steps_GELU.pth"
in_channels = 5
projection_dim = 128
batch_size = 1
num_workers = 4
max_batches = 300 #900
n_steps = 10 #at least 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load Datasets
# ------------------------------
print("🔍 Loading datasets...")
test_dataset = xr.open_zarr(f"{directory}/data/test_dataset_normalized_by_member_weekly_adjacent_timestep.zarr", consolidated=True)
static_ds = xr.open_zarr(f"{directory}/data/static_input.zarr", consolidated=False)
static_input_tensor = torch.tensor(static_ds["static_input"].values, dtype=torch.float32).unsqueeze(0)

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

# ------------------------------
# (Optional) Save or Analyze Results
# ------------------------------
print(f"✅ Done. Collected {len(results)} results.")
