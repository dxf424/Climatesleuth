import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import dask

# Define the variable to extract from each dataset
variable_name = 'TGCLDLWP'

# Configure Dask to use multi-threaded scheduling for parallel operations
dask.config.set(scheduler='threads')

def load_member(member, scheme):
    """
    Load a single member of an ensemble dataset for a given microphysics scheme.

    Args:
        member (int): Member index (1-based).
        scheme (str): Microphysics scheme identifier (e.g., 'ml' or 'kk').

    Returns:
        xarray.DataArray: The specified variable as a DataArray.
    """
    # Construct the file path using the scheme and member index
    path = (
        f'/glade/work/fanda/miles-climatesleuth/microphysics_analysis/daily_time_series/'
        f'{scheme}/PPE_100_{scheme}.{member:02}_daily_TGCLDLWP.nc'
    )
    
    # Open the NetCDF file using Dask chunks for parallel access
    ds = xr.open_dataset(path, chunks={'time': 1})
    
    # Return only the target variable (e.g., TGCLDLWP)
    return ds[variable_name]

def load_ensemble_data(scheme: str, num_members: int) -> xr.DataArray:
    """
    Load an ensemble dataset in parallel using multiple processes.

    Args:
        scheme (str): Microphysics scheme name ('ml' or 'kk').
        num_members (int): Number of ensemble members to load.

    Returns:
        xarray.DataArray: Concatenated DataArray with dimension 'member'.
    """
    # Use ProcessPoolExecutor to load members in parallel (up to 8 workers)
    with ProcessPoolExecutor(max_workers=min(8, num_members)) as executor:
        # Launch parallel loading for all member indices
        members = list(
            executor.map(
                load_member,
                range(1, num_members + 1),
                [scheme] * num_members  # Pass the same scheme to all calls
            )
        )
    
    # Concatenate all member DataArrays along the 'member' dimension
    return xr.concat(members, dim='member')
