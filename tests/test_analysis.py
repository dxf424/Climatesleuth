import torch
import xarray as xr
from climatesleuth.analysis.integrated_gradient import compute_integrated_gradients
#from analysis.backward_optimization import compute_integrated_gradients
from climatesleuth.models import MeanPooledVGG
from climatesleuth.datasets import XarrayBalancedBatchDataset
from torch.utils.data import DataLoader

def test_compute_integrated_gradients_runs():
    # Dummy dataset (you can mock or use small real samples)
    ds = xr.Dataset({
        "data": (("pairs", "member", "lat", "lon"), torch.rand(4, 2, 32, 32).numpy()),
        "labels": (("pairs",), [0, 1, 0, 1]),
        "week": (("pairs",), ["0001-01", "0001-02", "0001-01", "0001-02"]),
        "member": (("pairs",), [0, 1, 2, 3])
    })

    static_input_tensor = torch.rand(1, 4, 32, 32)
    test_dataset = ds
    dataset = XarrayBalancedBatchDataset(test_dataset, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MeanPooledVGG(in_channels=5, projection_dim=128)
    
    results = compute_integrated_gradients(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,
        static_input_tensor=static_input_tensor,
        device=torch.device("cpu"),
        max_batches=1
    )

    assert isinstance(results, list)
    assert "mean_gradient_ML" in results[0]
