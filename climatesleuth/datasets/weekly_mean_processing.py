import xarray as xr

def compute_weekly_datasets(ml_daily, kk_daily):
    """
    Convert daily ML and KK datasets into weekly-averaged datasets.

    Args:
        ml_daily (xr.DataArray): ML daily dataset with 'time' dimension.
        kk_daily (xr.DataArray): KK daily dataset with 'time' dimension.

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: Weekly-averaged ML and KK datasets.
    """
    # Remove the first month (e.g., spin-up) by slicing from February 1
    ml_daily = ml_daily.sel(time=slice("0001-02-01", "0003-12-31"))
    kk_daily = kk_daily.sel(time=slice("0001-02-01", "0003-12-31"))

    # Create a new 'week' coordinate based on the year and week number
    ml_daily = ml_daily.assign_coords(week=ml_daily['time'].dt.strftime('%Y-%W'))
    kk_daily = kk_daily.assign_coords(week=kk_daily['time'].dt.strftime('%Y-%W'))

    # Group by 'week' and compute the mean for each week
    ml_weekly = ml_daily.groupby('week').mean()
    kk_weekly = kk_daily.groupby('week').mean()

    return ml_weekly, kk_weekly

def normalize_weekly_datasets(ml_weekly, kk_weekly):
    """
    Normalize ML and KK weekly datasets by their own mean and std, split by data phase.

    Args:
        ml_weekly (xr.DataArray): Weekly-averaged ML dataset.
        kk_weekly (xr.DataArray): Weekly-averaged KK dataset.

    Returns:
        dict: Dictionary with normalized splits:
              - "ml_train", "ml_val", "ml_test"
              - "kk_train", "kk_val", "kk_test"
    """
    # Define splits by member index ranges
    slices = {
        'train': slice(0, 60),   # First 60 members for training
        'val': slice(60, 80),    # Next 20 for validation
        'test': slice(80, 100)   # Last 20 for testing
    }

    normalized = {}
    for split, s in slices.items():
        # Select member slice
        ml = ml_weekly.sel(member=s)
        kk = kk_weekly.sel(member=s)

        # Normalize each slice using Z-score (mean and std over 'week')
        ml_norm = (ml - ml.mean(dim=["week"])) / ml.std(dim=["week"])
        kk_norm = (kk - kk.mean(dim=["week"])) / kk.std(dim=["week"])

        # Fill NaNs (e.g., from constant values or missing weeks) with zero
        normalized[f"ml_{split}"] = ml_norm.fillna(0)
        normalized[f"kk_{split}"] = kk_norm.fillna(0)

    return normalized
