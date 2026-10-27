import climatesleuth
from climatesleuth.evaluation import evaluate_contrastive_model
from climatesleuth.models import MeanPooledResNet18, MeanPooledVGG
from climatesleuth.datasets import XarrayBalancedBatchPrefetcher_evaluation, BatchForecastLenDataLoader, XarrayBalancedBatchPrefetcher_evaluation_monthly
import xarray as xr
import torch
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run backward optimization and evaluation.")
    parser.add_argument("--directory", type=str, required=True,
                    help="Path to data and model directory.")
    args = parser.parse_args()

    # Use the directory from command-line arguments
    directory = args.directory

    # Choose your model
    model_class = MeanPooledVGG  # or MeanPooledResNet18
    train_dataset = xr.open_zarr(f"{directory}/data/train_dataset_normalized_by_member_monthly_adjacent_timestep_TOTAL_PRECIP.zarr", consolidated=True)
    val_dataset   = xr.open_zarr(f"{directory}/data/val_dataset_normalized_by_member_monthly_adjacent_timestep_TOTAL_PRECIP.zarr", consolidated=True)
    test_dataset  = xr.open_zarr(f"{directory}/data/test_dataset_normalized_by_member_monthly_adjacent_timestep_TOTAL_PRECIP.zarr", consolidated=True)

    # Load and prepare static input
    static_ds = xr.open_zarr(f"{directory}/data/static_input.zarr", consolidated=False)
    static_input_tensor = torch.tensor(static_ds["static_input"].values, dtype=torch.float32).unsqueeze(0)
    static_repeated = static_input_tensor.repeat(2, 1, 1, 1)

    evaluate_contrastive_model(
        model_class=model_class,
        save_path=f"{directory}/model/contrastive_model_vgg_seed_42_rank_0_100_accumulation_steps_GELU_total_precip.pth",
        data_structure='monthly',
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        static_repeated=static_repeated,
        batch_size=1,
        num_workers=4,
        in_channels=5,
        projection_dim=128
    )
    
if __name__ == "__main__":
    main()
