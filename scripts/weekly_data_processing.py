import time
from datasets import (
    load_ensemble_data,
    compute_weekly_datasets,
    normalize_weekly_datasets,
    MemberPairDataset
)

def main():
    start = time.time()
    ml = load_ensemble_data('ml', 100)
    kk = load_ensemble_data('kk', 100)

    ml_weekly, kk_weekly = compute_weekly_datasets(ml, kk)
    norm = normalize_weekly_datasets(ml_weekly, kk_weekly)

    train_ds = MemberPairDataset(norm["ml_train"], norm["kk_train"])
    val_ds = MemberPairDataset(norm["ml_val"], norm["kk_val"])
    test_ds = MemberPairDataset(norm["ml_test"], norm["kk_test"])

    train_ds.save_to_zarr("train_dataset.zarr")
    val_ds.save_to_zarr("val_dataset.zarr")
    test_ds.save_to_zarr("test_dataset.zarr")
    print(f"Done in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
