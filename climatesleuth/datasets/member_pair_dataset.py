import random
import torch
import xarray as xr
import dask.array as da
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class MemberPairDataset(Dataset):
    def __init__(self, ml_data=None, kk_data=None, num_pairs=30, random_seed=42, num_workers=8, preloaded_data=None):
        """
        Create a PyTorch-compatible dataset for contrastive learning using adjacent time step pairs
        from ML and KK ensemble data.

        Args:
            ml_data (xr.DataArray): Normalized ML dataset.
            kk_data (xr.DataArray): Normalized KK dataset.
            num_pairs (int): Number of pairs to generate (not currently used directly).
            random_seed (int): Seed for reproducibility.
            num_workers (int): Number of threads for parallel pair generation.
            preloaded_data (dict): Preloaded xarray dataset with 'data' and 'labels' keys (optional).
        """
        self.num_pairs = num_pairs
        self.num_workers = num_workers

        # Load from pre-generated Zarr data
        if preloaded_data:
            self.data = preloaded_data["data"]
            self.labels = preloaded_data["labels"]
        else:
            # Ensure required inputs are provided
            if ml_data is None or kk_data is None:
                raise ValueError("ml_data and kk_data must be provided if not loading from a file.")

            random.seed(random_seed)

            self.ml_data = ml_data  # Shape: (member, week, lat, lon)
            self.kk_data = kk_data

            # Generate all adjacent week pairs for ML and KK data
            self.ml_pairs, self.ml_meta = self._generate_member_pairs(self.ml_data)
            self.kk_pairs, self.kk_meta = self._generate_member_pairs(self.kk_data)

            # Concatenate all pairs and labels (0 for ML, 1 for KK)
            self.data = self.ml_pairs + self.kk_pairs
            self.labels = [0] * len(self.ml_pairs) + [1] * len(self.kk_pairs)
            self.meta = self.ml_meta + self.kk_meta  # (member_idx, week)

            # Store member and week metadata as xarray DataArrays
            self.members = xr.DataArray(
                [m for m, _ in self.meta], dims="pairs", coords={"pairs": range(len(self.meta))}
            )
            self.weeks = xr.DataArray(
                [w for _, w in self.meta], dims="pairs", coords={"pairs": range(len(self.meta))}
            )

    def _generate_member_pairs(self, data):
        """
        Generate adjacent time-step pairs for each member.

        Args:
            data (xr.DataArray): Ensemble data.

        Returns:
            list of DataArrays: Paired samples.
            list of tuples: Metadata (member index, week string).
        """
        num_members = data.sizes["member"]
        num_time_steps = data.sizes["week"]

        pairs = []
        meta = []

        # Use threads to speed up pair generation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._create_adjacent_time_pair, data, m, t)
                for m in range(num_members)
                for t in range(num_time_steps - 1)
            ]
            for future in futures:
                pair, info = future.result()
                pairs.append(pair)
                meta.append(info)

        return pairs, meta

    def _create_adjacent_time_pair(self, data, member_idx, time_idx):
        """
        Create a pair of adjacent time steps for a single member.

        Args:
            data (xr.DataArray): Data source.
            member_idx (int): Index of the ensemble member.
            time_idx (int): Index of the current time step.

        Returns:
            xr.DataArray: Concatenated pair of time steps.
            tuple: (member index, starting week string).
        """
        week1 = data["week"].values[time_idx]
        week2 = data["week"].values[time_idx + 1]

        # Stack adjacent time steps along a new 'member' dimension
        pair = xr.concat([
            data.sel(member=member_idx, week=week1),
            data.sel(member=member_idx, week=week2)
        ], dim="member")

        return pair, (member_idx, week1)

    def __len__(self):
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample and label, loading the data if necessary.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input tensor and label.
        """
        x = torch.tensor(self.data.isel(pairs=idx).load().values, dtype=torch.float32)
        label = torch.tensor(int(self.labels.isel(pairs=idx).load().values), dtype=torch.long)
        return x, label

    def save_to_zarr(self, filepath):
        """
        Save the dataset to disk in Zarr format.

        Args:
            filepath (str): Output Zarr path.
        """
        # Concatenate list of pairs into a single xarray DataArray
        data = xr.concat(self.data, dim="pairs", join="override")

        # Create DataArray for labels
        labels = xr.DataArray(self.labels, dims="pairs", name="labels")

        # Add metadata coordinates
        data = data.assign_coords(pairs=("pairs", range(data.sizes["pairs"])))
        data = data.assign_coords(member=("pairs", self.members.data))
        data = data.assign_coords(week=("pairs", self.weeks.data))
        labels = labels.assign_coords(pairs=("pairs", range(labels.sizes["pairs"])))

        # Define chunking strategy
        chunk_sizes = {"pairs": 1, "member": 1, "lat": 192, "lon": 288}

        # Create dataset and write to Zarr
        dataset = xr.Dataset({
            "data": data.chunk(chunk_sizes),
            "labels": labels
        })
        dataset.to_zarr(filepath, mode="w", consolidated=True)
        print(f"Dataset saved to {filepath} with week/member coordinates.")

    @classmethod
    def load_from_zarr(cls, filepath):
        """
        Load a pre-saved Zarr dataset.

        Args:
            filepath (str): Path to Zarr directory.

        Returns:
            MemberPairDataset: Initialized dataset object.
        """
        dataset = xr.open_zarr(filepath)
        preloaded_data = {
            "data": dataset["data"],
            "labels": dataset["labels"]
        }
        return cls(preloaded_data=preloaded_data)

class Monthly_MemberPairDataset(Dataset):
    def __init__(self, ml_data=None, kk_data=None, num_workers=8, random_seed=42, preloaded_data=None):
        self.num_workers = num_workers

        if preloaded_data:
            self.data = preloaded_data["data"]
            self.labels = preloaded_data["labels"]
        else:
            if ml_data is None or kk_data is None:
                raise ValueError("ml_data and kk_data must be provided if not loading from file")

            self.ml_data = ml_data  # xarray.Dataset: (month, member, lat, lon) with many vars
            self.kk_data = kk_data

            random.seed(random_seed)

            self.ml_pairs, self.ml_meta = self._generate_member_pairs(self.ml_data)
            self.kk_pairs, self.kk_meta = self._generate_member_pairs(self.kk_data)

            self.data = self.ml_pairs + self.kk_pairs
            self.labels = [0] * len(self.ml_pairs) + [1] * len(self.kk_pairs)
            self.meta = self.ml_meta + self.kk_meta

            self.members = xr.DataArray([m for m, _ in self.meta], dims="pairs")
            self.months = xr.DataArray([mo for _, mo in self.meta], dims="pairs")

    def _generate_member_pairs(self, data):
        num_members = data.sizes["member"]
        num_time_steps = data.sizes["month"]

        pairs = []
        meta = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for m in range(num_members):
                for t in range(num_time_steps - 1):
                    futures.append(executor.submit(self._create_pair, data, m, t))

            for future in futures:
                pair, info = future.result()
                pairs.append(pair)
                meta.append(info)

        return pairs, meta

    def _create_pair(self, data, member_idx, month_idx):
        # Direct index access using .isel
        slice1 = data.isel(month=month_idx, member=member_idx)
        slice2 = data.isel(month=month_idx + 1, member=member_idx)
        # Stack each variable across a new 'var' dim
        vstack1 = xr.concat([slice1[v] for v in data.data_vars], dim="var")
        vstack2 = xr.concat([slice2[v] for v in data.data_vars], dim="var")

        # Combine into pair along a 'pair' dimension
        pair = xr.concat([vstack1, vstack2], dim="pair")

        # Still record the actual month string (optional metadata)
        m1_label = str(data["month"].values[month_idx])
        return pair, (member_idx, m1_label)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx].load().values, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def save_to_zarr(self, filepath):
        data = xr.concat(self.data, dim="pairs")
        data = data.assign_coords(pairs=("pairs", range(len(self.data))))
        labels = xr.DataArray(self.labels, dims="pairs", name="labels")

        dataset = xr.Dataset({
            "data": data.chunk({"pairs": 1}),
            "labels": labels
        })
        dataset.to_zarr(filepath, mode="w", consolidated=True)

    @classmethod
    def load_from_zarr(cls, filepath):
        ds = xr.open_zarr(filepath)
        return cls(preloaded_data={"data": ds["data"], "labels": ds["labels"]})
    
class XarrayBalancedBatchDataset(Dataset):
    def __init__(self, xarray_ds, shuffle=True):
        """
        Dataset that samples 10 label=0 and 10 label=1 samples
        for each (week, member) combination.
        """
        self.data = xarray_ds
        self.labels = self.data["labels"].values
        self.weeks = self.data["week"].values
        self.members = self.data["member"].values

        # Create list of unique (week, member) pairs
        week_member_pairs = list(zip(self.weeks, self.members))
        self.unique_pairs = list(set(week_member_pairs))

        # Shuffle for randomness
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(self.unique_pairs)

    def __len__(self):
        return len(self.unique_pairs)

    def __getitem__(self, idx):
        """
        Returns 20 indices from the same (week, member), balanced by label.
        """
        week_value, member_value = self.unique_pairs[idx]

        # Filter for the current (week, member)
        mask = (self.weeks == week_value) & (self.members == member_value)

        # Separate by labels
        zero_indices = np.flatnonzero((self.labels == 0) & mask)
        one_indices = np.flatnonzero((self.labels == 1) & mask)

        '''
        if len(zero_indices) < 10 or len(one_indices) < 10:
            raise ValueError(f"Not enough samples for (week={week_value}, member={member_value}): "
                             f"{len(zero_indices)} zeros, {len(one_indices)} ones")

        selected_zero = np.random.choice(zero_indices, 10, replace=False)
        selected_one = np.random.choice(one_indices, 10, replace=False)
        '''

        selected_indices = np.concatenate([zero_indices, one_indices])
        return selected_indices
    
class XarrayBalancedBatchDataset_monthly(Dataset):
    def __init__(self, xarray_ds, shuffle=True):
        """
        Dataset that samples 10 label=0 and 10 label=1 samples
        for each (month, member) combination.
        """
        self.data = xarray_ds
        self.labels = self.data["labels"].values
        self.months = self.data["month"][:,0].values
        self.members = self.data["member"].values

        # Create list of unique (month, member) pairs
        month_member_pairs = list(zip(self.months, self.members))
        self.unique_pairs = list(set(month_member_pairs))

        # Shuffle for randomness
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(self.unique_pairs)

    def __len__(self):
        return len(self.unique_pairs)

    def __getitem__(self, idx):
        """
        Returns 20 indices from the same (month, member), balanced by label.
        """
        month_value, member_value = self.unique_pairs[idx]

        # Filter for the current (month, member)
        mask = (self.months == month_value) & (self.members == member_value)

        # Separate by labels
        zero_indices = np.flatnonzero((self.labels == 0) & mask)
        one_indices = np.flatnonzero((self.labels == 1) & mask)
        selected_indices = np.concatenate([zero_indices, one_indices])
        return selected_indices

