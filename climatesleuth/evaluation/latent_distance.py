"""Latent-space distance utilities.

These helpers turn contrastive-learning embeddings into scalar distances that can
be used for model evaluation or calibration. They intentionally operate on NumPy
arrays or pandas DataFrames so they are independent of any one training script.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def _as_2d(array: np.ndarray) -> np.ndarray:
    """Return a float 2-D array and fail early for accidental bad shapes."""
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("Expected a 1-D embedding or 2-D embedding matrix")
    return array


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Calculate row-wise cosine distance ``1 - cosine_similarity``.

    Args:
        a: Embedding vector or matrix with shape ``(n, features)``.
        b: Embedding vector or matrix with shape ``(n, features)`` or one row
            that will be broadcast against ``a``.
        eps: Small value to avoid division by zero for degenerate embeddings.

    Returns:
        One distance per row after NumPy broadcasting.
    """
    a = _as_2d(a)
    b = _as_2d(b)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - numerator / np.maximum(denominator, eps)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate row-wise Euclidean distance between latent embeddings."""
    a = _as_2d(a)
    b = _as_2d(b)
    return np.linalg.norm(a - b, axis=1)


def latent_distance(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Calculate row-wise latent distance using a named metric.

    Supported metrics are ``"cosine"`` and ``"euclidean"``.
    """
    metric = metric.lower()
    if metric == "cosine":
        return cosine_distance(a, b)
    if metric in {"euclidean", "l2"}:
        return euclidean_distance(a, b)
    raise ValueError("metric must be 'cosine' or 'euclidean'")


def distance_to_reference(
    embeddings: np.ndarray,
    reference: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Calculate distance from each embedding row to one reference embedding."""
    embeddings = _as_2d(embeddings)
    reference = _as_2d(reference)
    if reference.shape[0] != 1:
        raise ValueError("reference must contain exactly one embedding")
    reference = np.repeat(reference, embeddings.shape[0], axis=0)
    return latent_distance(embeddings, reference, metric=metric)


def monthly_latent_distance(
    model_embeddings: pd.DataFrame,
    reference_embeddings: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    member_column: str = "member",
    month_column: str = "month",
    metric: str = "cosine",
) -> pd.DataFrame:
    """Calculate member-by-month latent distance to a monthly reference.

    Args:
        model_embeddings: DataFrame with one row per model member/month.
        reference_embeddings: DataFrame with one row per reference month. This is
            usually the observed monthly climatology embedded by the trained
            contrastive model.
        feature_columns: Latent feature columns. If omitted, all numeric columns
            except member/month are used.
        member_column: Column containing member IDs in ``model_embeddings``.
        month_column: Column containing month IDs in both DataFrames.
        metric: ``"cosine"`` or ``"euclidean"``.

    Returns:
        DataFrame with ``member``, ``month``, and ``latent_distance`` columns.
    """
    if feature_columns is None:
        excluded = {member_column, month_column}
        feature_columns = [c for c in model_embeddings.select_dtypes(include=[np.number]).columns if c not in excluded]
    feature_columns = list(feature_columns)

    ref = reference_embeddings.set_index(month_column)
    rows = []
    for _, row in model_embeddings.iterrows():
        month = row[month_column]
        if month not in ref.index:
            raise KeyError("Reference embeddings are missing month %s" % month)
        a = row[feature_columns].to_numpy(dtype=float)
        b = ref.loc[month, feature_columns].to_numpy(dtype=float)
        rows.append({
            member_column: row[member_column],
            month_column: month,
            "latent_distance": float(latent_distance(a, b, metric=metric)[0]),
        })
    return pd.DataFrame(rows)


def summarize_member_distance(
    monthly_distances: pd.DataFrame,
    member_column: str = "member",
    distance_column: str = "latent_distance",
) -> pd.DataFrame:
    """Summarize monthly latent distances to one mean distance per member."""
    return (
        monthly_distances.groupby(member_column, as_index=False)
        .agg(
            mean_latent_distance=(distance_column, "mean"),
            std_latent_distance=(distance_column, "std"),
            min_latent_distance=(distance_column, "min"),
            max_latent_distance=(distance_column, "max"),
            n_months=(distance_column, "count"),
        )
        .sort_values(member_column)
    )
