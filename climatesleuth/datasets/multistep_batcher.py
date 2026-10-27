import random
import torch
import numpy as np
from torch.utils.data import Dataset
from multiprocessing import Process, Queue, Event, Manager
from threading import Thread
import time
import signal


class XarrayBalancedBatchPrefetcher(Dataset):
    """
    A PyTorch-compatible dataset that asynchronously prefetches balanced batches
    (by label) from an xarray-based Zarr dataset, with DDP-aware sharding.
    """

    def __init__(self, xarray_ds, static_repeated, shuffle=True, num_workers=4,
                 prefetch_factor=8, rank=0, world_size=1):
        self.data = xarray_ds
        self.labels = self.data["labels"].values
        self.weeks = self.data["week"].values
        self.members = self.data["member"].values
        self.static_repeated = static_repeated
        self.rank = rank

        # Shard unique (week, member) pairs across DDP ranks
        all_pairs = list(set(zip(self.weeks, self.members)))
        all_pairs.sort()
        self.unique_pairs = all_pairs[rank::world_size]
        print(f"[Rank {rank}] total pairs: {len(self.unique_pairs)}")

        if shuffle:
            np.random.seed(42 + rank)
            np.random.shuffle(self.unique_pairs)

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = Queue(maxsize=prefetch_factor)
        self.stop_event = Event()
        self.manager = Manager()
        self.results = self.manager.dict()

        # Handle termination cleanly
        signal.signal(signal.SIGTERM, self.handle_signal)

        # Launch prefetching thread
        self.prefetch_thread = Thread(target=self.prefetch_batches, daemon=True)
        self.prefetch_thread.start()

    def set_epoch(self, epoch):
        """
        Reset prefetching logic for a new training epoch (DDP-compatible).
        """
        print(f"[Rank {self.rank}] set_epoch({epoch}) called")

        # Stop and cleanup current prefetch thread
        self.stop_event.set()
        if hasattr(self, "prefetch_thread") and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=5)

        # Reset infrastructure
        self.stop_event = Event()
        self.prefetch_queue = Queue(maxsize=self.prefetch_factor)
        try:
            self.manager.shutdown()
        except Exception as e:
            print(f"[Rank {self.rank}] Manager shutdown warning: {e}")
        self.manager = Manager()
        self.results = self.manager.dict()

        # Start new prefetch thread
        self.prefetch_thread = Thread(target=self.prefetch_batches, daemon=True)
        self.prefetch_thread.start()

    def __len__(self):
        return len(self.unique_pairs)

    def __getitem__(self, _):
        return self.prefetch_queue.get()

    def handle_signal(self, signum, frame):
        print("Received termination signal. Stopping prefetching...")
        self.stop_event.set()

    def prefetch_batches(self):
        idx = 0
        try:
            while not self.stop_event.is_set():
                if not self.prefetch_queue.full():
                    chunk = self.unique_pairs[idx:idx + self.num_workers]
                    if self.results is None:
                        self.manager = Manager()
                        self.results = self.manager.dict()
                    else:
                        self.results.clear()

                    # Launch multiprocessing workers
                    processes = [
                        Process(target=self.worker_process, args=(i, pair, self.results))
                        for i, pair in enumerate(chunk)
                    ]
                    for p in processes: p.start()
                    for p in processes: p.join()

                    # Collect results
                    for i in range(len(chunk)):
                        if self.results.get(i) is not None:
                            self.prefetch_queue.put(self.results[i])

                    idx += self.num_workers
                else:
                    time.sleep(0.05)
        except Exception as e:
            print(f"Prefetching error: {e}")

    def worker_process(self, k, pair, result_dict):
        try:
            week_value, member_value = pair
            mask = (self.weeks == week_value) & (self.members == member_value)
            zero_indices = np.flatnonzero((self.labels == 0) & mask)
            one_indices = np.flatnonzero((self.labels == 1) & mask)

            selected_indices = np.concatenate([zero_indices, one_indices])
            selected_indices = selected_indices[selected_indices < self.data.sizes["pairs"]]
            if len(selected_indices) == 0:
                result_dict[k] = None
                return

            data_batch = self.data["data"].isel(pairs=selected_indices).load()
            label_batch = torch.tensor(self.labels[selected_indices], dtype=torch.long)

            x1 = torch.from_numpy(data_batch.isel(member=0).values).float().unsqueeze(1)
            x2 = torch.from_numpy(data_batch.isel(member=1).values).float().unsqueeze(1)

            x1_combined = torch.cat([x1, self.static_repeated], dim=1)
            x2_combined = torch.cat([x2, self.static_repeated], dim=1)

            result_dict[k] = {"x1": x1_combined, "x2": x2_combined, "x1_labels": label_batch, "x2_labels": label_batch}
        except Exception as e:
            print(f"Worker error for pair {pair}: {e}")
            result_dict[k] = None

    def __del__(self):
        """
        Graceful cleanup of multiprocessing/threading resources.
        """
        try:
            print(f"[Rank {getattr(self, 'rank', '?')}] Cleaning up...")
            self.stop_event.set()

            if hasattr(self, "prefetch_thread") and self.prefetch_thread.is_alive():
                self.prefetch_thread.join(timeout=5)

            if hasattr(self, "manager"):
                try:
                    self.manager.shutdown()
                except Exception as e:
                    print(f"[Rank {getattr(self, 'rank', '?')}] Manager shutdown error: {e}")

            if hasattr(self, "prefetch_queue"):
                while not self.prefetch_queue.empty():
                    try:
                        self.prefetch_queue.get_nowait()
                    except Exception:
                        break

            print(f"[Rank {getattr(self, 'rank', '?')}] Cleanup complete.")
        except Exception as e:
            print(f"[Rank {getattr(self, 'rank', '?')}] Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    @property
    def forecast_len(self):
        return 0

    def batches_per_epoch(self):
        return len(self)


class BatchForecastLenDataLoader:
    """
    A wrapper over the dataset that respects a forecast horizon (if needed).
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.forecast_len = dataset.forecast_len + 1

    def __iter__(self):
        dataset_iter = iter(self.dataset)
        for _ in range(len(self)):
            yield next(dataset_iter)

    def __len__(self):
        if hasattr(self.dataset, "batches_per_epoch"):
            return self.dataset.batches_per_epoch() * self.forecast_len
        else:
            return len(self.dataset) * self.forecast_len

class XarrayBalancedBatchPrefetcher_evaluation(Dataset):
    def __init__(self, xarray_ds, static_repeated, shuffle=True, num_workers=4, prefetch_factor=8):
        self.data = xarray_ds
        self.labels = self.data["labels"].values
        self.weeks = self.data["week"].values
        self.members = self.data["member"].values
        self.static_repeated = static_repeated
        #self.device = device

        self.unique_pairs = list(set(zip(self.weeks, self.members)))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(self.unique_pairs)

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = Queue(maxsize=prefetch_factor)
        self.stop_event = Event()
        self.manager = Manager()
        self.results = self.manager.dict()

        signal.signal(signal.SIGTERM, self.handle_signal)

        self.prefetch_thread = Thread(target=self.prefetch_batches, daemon=True)
        self.prefetch_thread.start()

    def __len__(self):
        return len(self.unique_pairs)

    def __getitem__(self, _):
        return self.prefetch_queue.get()

    def handle_signal(self, signum, frame):
        print("Received termination signal. Stopping prefetching...")
        self.stop_event.set()

    def prefetch_batches(self):
        idx = 0
        try:
            while not self.stop_event.is_set() and idx < len(self):
                if not self.prefetch_queue.full():
                    chunk = self.unique_pairs[idx:idx + self.num_workers]
                    self.results.clear()
                    processes = [
                        Process(target=self.worker_process, args=(i, pair, self.results))
                        for i, pair in enumerate(chunk)
                    ]
                    for p in processes:
                        p.start()
                    for p in processes:
                        p.join()

                    for i in range(len(chunk)):
                        if self.results.get(i) is not None:
                            self.prefetch_queue.put(self.results[i])
                    idx += self.num_workers
                else:
                    time.sleep(0.05)
        except Exception as e:
            print(f"Prefetching error: {e}")

    def worker_process(self, k, pair, result_dict):
        try:
            week_value, member_value = pair
            mask = (self.weeks == week_value) & (self.members == member_value)
            zero_indices = np.flatnonzero((self.labels == 0) & mask)
            one_indices = np.flatnonzero((self.labels == 1) & mask)

            selected_indices = np.concatenate([zero_indices, one_indices])
            selected_indices = selected_indices[selected_indices < self.data.sizes["pairs"]]
            if len(selected_indices) == 0:
                result_dict[k] = None
                return

            data_batch = self.data["data"].isel(pairs=selected_indices).load()
            label_batch = torch.tensor(self.labels[selected_indices], dtype=torch.long)

            x1 = torch.from_numpy(data_batch.isel(member=0).values).float().unsqueeze(1)#.to(self.device, non_blocking=True)
            x2 = torch.from_numpy(data_batch.isel(member=1).values).float().unsqueeze(1)#.to(self.device, non_blocking=True)

            x1_combined = torch.cat([x1, self.static_repeated], dim=1)
            x2_combined = torch.cat([x2, self.static_repeated], dim=1)

            result_dict[k] = {"x1": x1_combined, "x2": x2_combined, "x1_labels": label_batch, "x2_labels": label_batch}

        except Exception as e:
            print(f"Worker error for pair {pair}: {e}")
            result_dict[k] = None

    def __del__(self):
        try:
            self.stop_event.set()
            if self.prefetch_thread.is_alive():
                self.prefetch_thread.join(timeout=2)
            self.manager.shutdown()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    @property
    def forecast_len(self):
        return 0

    def batches_per_epoch(self):
        return len(self) // self.num_workers
    
class XarrayBalancedBatchPrefetcher_evaluation_monthly(Dataset):
    def __init__(self, xarray_ds, static_repeated, shuffle=True, num_workers=4, prefetch_factor=8):
        self.data = xarray_ds
        self.labels = self.data["labels"].values
        self.months = self.data["month"].values[:, 0]  # use month[:, 0] as base time
        self.members = self.data["member"].values
        self.static_repeated = static_repeated

        all_pairs = list(set(zip(self.months, self.members)))
        all_pairs.sort()
        self.unique_pairs = all_pairs
        print(len(self.unique_pairs))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(self.unique_pairs)

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = Queue(maxsize=prefetch_factor)
        self.stop_event = Event()
        self.manager = Manager()
        self.results = self.manager.dict()

        signal.signal(signal.SIGTERM, self.handle_signal)

        self.prefetch_thread = Thread(target=self.prefetch_batches, daemon=True)
        self.prefetch_thread.start()

    def set_epoch(self, epoch):
        print(f"set_epoch({epoch}) called")

        self.stop_event.set()
        if hasattr(self, "prefetch_thread") and self.prefetch_thread is not None:
            if self.prefetch_thread.is_alive():
                self.prefetch_thread.join(timeout=5)

        self.stop_event = Event()
        self.prefetch_queue = Queue(maxsize=self.prefetch_factor)

        try:
            if hasattr(self, "manager") and self.manager is not None:
                self.manager.shutdown()
        except Exception as e:
            print(f"Warning: failed to shutdown manager: {e}")

        self.manager = Manager()
        self.results = self.manager.dict()

        self.prefetch_thread = Thread(target=self.prefetch_batches, daemon=True)
        self.prefetch_thread.start()

    def __len__(self):
        return len(self.unique_pairs)

    def __getitem__(self, _):
        return self.prefetch_queue.get()

    def handle_signal(self, signum, frame):
        print("Received termination signal. Stopping prefetching...")
        self.stop_event.set()

    def prefetch_batches(self):
        idx = 0
        try:
            while not self.stop_event.is_set():
                if not self.prefetch_queue.full():
                    chunk = self.unique_pairs[idx:idx + self.num_workers]
                    if self.results is None:
                        if self.manager is None:
                            self.manager = Manager()
                        self.results = self.manager.dict()
                    else:
                        self.results.clear()
                    processes = [
                        Process(target=self.worker_process, args=(i, pair, self.results))
                        for i, pair in enumerate(chunk)
                    ]
                    for p in processes:
                        p.start()
                    for p in processes:
                        p.join()

                    for i in range(len(chunk)):
                        if self.results.get(i) is not None:
                            self.prefetch_queue.put(self.results[i])
                    idx += self.num_workers
                else:
                    time.sleep(0.05)
        except Exception as e:
            print(f"Prefetching error: {e}")
    def worker_process(self, k, pair, result_dict):
        try:
            month_value, member_value = pair
            mask = (self.months == month_value) & (self.members == member_value)
            zero_indices = np.flatnonzero((self.labels == 0) & mask)
            one_indices = np.flatnonzero((self.labels == 1) & mask)

            selected_indices = np.concatenate([zero_indices, one_indices])
            if len(selected_indices) == 0:
                result_dict[k] = None
                return

            data_batch = self.data["data"].isel(pairs=selected_indices).load()
            label_batch = torch.tensor(self.labels[selected_indices], dtype=torch.long)

            x1 = torch.from_numpy(data_batch[:, 0].values).float()
            x2 = torch.from_numpy(data_batch[:, 1].values).float()

            x1_combined = torch.cat([x1, self.static_repeated], dim=1)
            x2_combined = torch.cat([x2, self.static_repeated], dim=1)

            result_dict[k] = {
                "x1": x1_combined,
                "x2": x2_combined,
                "x1_labels": label_batch,
                "x2_labels": label_batch,
            }

        except Exception as e:
            print(f"Worker error for pair {pair}: {e}")
            result_dict[k] = None
        
    def __del__(self):
        try:
            print("Cleaning up...")
            self.stop_event.set()
            if hasattr(self, "prefetch_thread") and self.prefetch_thread.is_alive():
                self.prefetch_thread.join(timeout=5)
            if hasattr(self, "manager"):
                try:
                    self.manager.shutdown()
                except Exception as e:
                    print(f"Manager shutdown error: {e}")
            if hasattr(self, "prefetch_queue"):
                while not self.prefetch_queue.empty():
                    try:
                        self.prefetch_queue.get_nowait()
                    except Exception:
                        break
            print("Cleanup complete.")
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    @property
    def forecast_len(self):
        return 0

    def batches_per_epoch(self):
        return len(self)

