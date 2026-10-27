import numpy as np
import xarray as xr
import torch
from scipy.interpolate import griddata

def normalize(data):
    """
    Apply standard Z-score normalization to a NumPy array.

    Args:
        data (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array with mean 0 and std 1.
    """
    return (data - np.mean(data)) / np.std(data)

def get_static_input_tensor(lat_target, lon_target) -> torch.Tensor:
    """
    Load and interpolate static features (e.g., elevation, land-sea mask) to match the target spatial grid.

    Args:
        lat_target (np.ndarray): 1D array of target latitudes.
        lon_target (np.ndarray): 1D array of target longitudes.

    Returns:
        torch.Tensor: Tensor of shape (1, 4, H, W), where the 4 channels are:
                      - normalized elevation
                      - land-sea mask
                      - normalized longitude
                      - normalized latitude
    """
    # Load static dataset containing elevation (Z) and land-sea mask (LSM)
    ds = xr.open_dataset("/glade/campaign/cisl/aiml/ksha/CREDIT/static_norm_old.nc")

    # Extract variables and coordinate grids from the static dataset
    Z = ds["Z_GDS4_SFC"].values       # 2D elevation field
    LSM = ds["LSM"].values            # 2D land-sea mask (categorical: 0 or 1)
    lat_static = ds["latitude"].values
    lon_static = ds["longitude"].values

    # Generate a 2D grid from the target latitude and longitude arrays
    lon_grid, lat_grid = np.meshgrid(lon_target, lat_target)

    # Flatten static coordinates into point list for interpolation
    points = np.array(np.meshgrid(lon_static, lat_static)).T.reshape(-1, 2)  # shape: (N, 2)

    # Interpolate elevation and land-sea mask onto the target grid
    Z_interp = griddata(points, Z.ravel(), (lon_grid, lat_grid), method="nearest")
    LSM_interp = griddata(points, LSM.ravel(), (lon_grid, lat_grid), method="nearest")

    # Stack all four channels:
    # - normalized elevation
    # - raw LSM (binary mask)
    # - normalized longitude
    # - normalized latitude
    static_input = np.stack([
        normalize(Z_interp),
        LSM_interp,
        normalize(lon_grid),
        normalize(lat_grid)
    ], axis=0)  # Resulting shape: (4, H, W)

    # Convert to a 4D PyTorch tensor and add a batch dimension: (1, 4, H, W)
    return torch.tensor(static_input, dtype=torch.float32).unsqueeze(0)
