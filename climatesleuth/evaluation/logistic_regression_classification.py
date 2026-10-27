import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from climatesleuth.datasets import XarrayBalancedBatchPrefetcher_evaluation, BatchForecastLenDataLoader, XarrayBalancedBatchPrefetcher_evaluation_monthly
from climatesleuth.evaluation import visualize_latent_features
from tqdm import tqdm

def extract_features(model, dataloader, device):
    features = []
    labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            x1 = batch["x1"].to(device, non_blocking=True)         # [N, C, H, W]
            x2 = batch["x2"].to(device, non_blocking=True)         # [N, C, H, W]
            labels1 = batch["x1_labels"].cpu().numpy()             # [N]
            labels2 = batch["x2_labels"].cpu().numpy()             # [N]

            # Feature extraction
            z1 = model(x1).cpu().numpy()
            z2 = model(x2).cpu().numpy()

            # Mean LWP proxy
            f1 = x1.mean(dim=(1, 2, 3)).cpu().numpy()
            f2 = x2.mean(dim=(1, 2, 3)).cpu().numpy()

            # Combine all
            features.extend(z1)
            features.extend(z2)
            labels.extend(labels1)
            labels.extend(labels2)

    return np.array(features), np.array(labels)


def evaluate_contrastive_model(
    model_class,
    save_path,
    data_structure,
    train_dataset, val_dataset, test_dataset, static_repeated,
    batch_size=1,
    num_workers=4,
    in_channels=5,
    projection_dim=128,
    return_model=True,
    return_feature=True,
    for_analysis=False
):
    """
    Run contrastive evaluation by extracting features from a pretrained model and training a logistic regression classifier.

    Args:
        model_class (nn.Module): Model class to initialize (e.g., MeanPooledResNet18 or MeanPooledVGG).
        save_path (str): Path to the saved model weights.
        static_input_tensor (torch.Tensor): Static variables to concatenate to input.
        train_dataset, val_dataset, test_dataset (xarray.Dataset): Datasets.
        batch_size (int): Dataloader batch size.
        num_workers (int): Dataloader worker threads.
        in_channels (int): Number of input channels (e.g., 5).
        projection_dim (int): Projection head output dimension.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and remove DDP prefixes
    model = model_class(in_channels=in_channels, projection_dim=projection_dim)
    state_dict = torch.load(save_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    # Initialize datasets
    if data_structure == 'monthly':
        train_dataset = XarrayBalancedBatchPrefetcher_evaluation_monthly(
            xarray_ds=train_dataset,
            static_repeated=static_repeated,
            shuffle=True,
            num_workers=4,
            prefetch_factor=4,
        )

        val_dataset = XarrayBalancedBatchPrefetcher_evaluation_monthly(
            xarray_ds=val_dataset,
            static_repeated=static_repeated,
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
        )

        test_dataset = XarrayBalancedBatchPrefetcher_evaluation_monthly(
            xarray_ds=test_dataset,
            static_repeated=static_repeated,
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
        )
    else:
        train_dataset = XarrayBalancedBatchPrefetcher_evaluation(
            xarray_ds=train_dataset,
            static_repeated=static_repeated,
            shuffle=True,
            num_workers=4,
            prefetch_factor=4,
        )

        val_dataset = XarrayBalancedBatchPrefetcher_evaluation(
            xarray_ds=val_dataset,
            static_repeated=static_repeated,
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
        )

        test_dataset = XarrayBalancedBatchPrefetcher_evaluation(
            xarray_ds=test_dataset,
            static_repeated=static_repeated,
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
        )

    # Wrap in batch loaders
    train_loader = BatchForecastLenDataLoader(train_dataset)
    val_loader = BatchForecastLenDataLoader(val_dataset)
    test_loader = BatchForecastLenDataLoader(test_dataset)
    
    print("📥 Extracting features...")
    X_test, y_test = extract_features(model, test_loader, device)#, test_dataset, static_input_tensor)
    
    if for_analysis==False:
        X_train, y_train = extract_features(model, train_loader, device)#, train_dataset, static_input_tensor)
        X_val, y_val = extract_features(model, val_loader, device)#, val_dataset, static_input_tensor)

        print("🎯 Training logistic regression classifier...")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["ML (Class 0)", "KK (Class 1)"])

        print(f"✅ Linear Evaluation Accuracy: {accuracy:.4f}")
        print("📊 Classification Report:\n", report)

        # Visualize using PCA (for faster results)
        Features, explained_ratio = visualize_latent_features(X_test, y_test, method="pca")
    
    return model, X_test, y_test#clf, accuracy, report
