"""Contrastive losses used by ClimateSleuth models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Symmetric normalized temperature-scaled cross entropy loss.

    This is the SimCLR-style NT-Xent objective used by the monthly radiation/TLWP
    training workflow. Given two batches of paired embeddings ``z_i`` and
    ``z_j``, the loss treats ``z_j[k]`` as the positive example for ``z_i[k]``
    and vice versa, while all other embeddings in the combined batch are
    negatives.

    Args:
        temperature: Positive scalar used to scale cosine-similarity logits.
        normalize: If True, L2-normalize embeddings before computing logits.

    Shape:
        z_i: ``(batch, embedding_dim)``
        z_j: ``(batch, embedding_dim)``

    Returns:
        Scalar tensor containing the mean cross-entropy loss over ``2 * batch``
        anchors.
    """

    def __init__(self, temperature: float = 0.5, normalize: bool = True) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.ndim != 2 or z_j.ndim != 2:
            raise ValueError("z_i and z_j must be rank-2 tensors: (batch, embedding_dim)")
        if z_i.shape != z_j.shape:
            raise ValueError(f"z_i and z_j must have the same shape, got {z_i.shape} and {z_j.shape}")

        batch_size = z_i.size(0)
        if batch_size < 2:
            raise ValueError("NTXentLoss requires at least two paired samples per batch")

        z = torch.cat([z_i, z_j], dim=0)
        if self.normalize:
            z = F.normalize(z, dim=1)

        logits = torch.matmul(z, z.T) / self.temperature
        self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, float("-inf"))

        targets = torch.arange(2 * batch_size, device=z.device)
        targets = (targets + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, targets)
