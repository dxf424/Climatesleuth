import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

plt.rcParams['font.size'] = 14

def visualize_latent_features(X, y, method="tsne", output_dir="figures", filename_prefix="latent"):
    """
    Visualizes the extracted features using t-SNE or PCA and saves the plot.
    
    Args:
        X: Feature matrix (latent embeddings) - shape [num_samples, feature_dim]
        y: Labels corresponding to the embeddings
        method: "tsne" or "pca"
        output_dir: Directory where plots will be saved
        filename_prefix: Prefix for saved filenames

    Returns:
        X_reduced: Reduced 2D features
        explained_ratio: (PCA only) Explained variance ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    if method == "tsne":
        print("Performing t-SNE...")
        X_reduced = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(X_normalized)
        explained_ratio = None
    elif method == "pca":
        print("Performing PCA...")
        pca_full = PCA(n_components=10)
        X_reduced_10d = pca_full.fit_transform(X_normalized)
        explained_ratio = pca_full.explained_variance_ratio_

        pca_2d = PCA(n_components=2)
        X_reduced = pca_2d.fit_transform(X_normalized)
        print("Explained variance ratio (2D):", pca_2d.explained_variance_ratio_)
    else:
        raise ValueError("Method should be either 'tsne' or 'pca'")

    # Plot latent space
    plt.figure(figsize=(12, 8))
    class_labels = {0: "TAU-ML", 1: "KK2000"}
    colors = ["blue", "red"]

    for label, color in zip(class_labels.keys(), colors):
        mask = (y == label)
        plt.scatter(
            X_reduced[mask, 0], X_reduced[mask, 1],
            edgecolors=color, facecolors='none', s=80, label=class_labels[label]
        )
    
    plt.title(f"Latent Space Visualization using {method.upper()}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    fig_path = os.path.join(output_dir, f"{filename_prefix}_{method}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent space plot to: {fig_path}")

    # Save explained variance plot for PCA
    if method == "pca" and explained_ratio is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(explained_ratio) + 1), explained_ratio)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.xticks(range(1, len(explained_ratio) + 1))
        plt.grid(True)
        plt.tight_layout()

        var_fig_path = os.path.join(output_dir, f"{filename_prefix}_pca_variance.png")
        plt.savefig(var_fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA variance plot to: {var_fig_path}")

    return X_reduced, explained_ratio
