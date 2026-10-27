#!/usr/bin/env python
"""Minimal sample for the channel-aware monthly radiation/TLWP model.

This script is intentionally small: real DDP data loading lives in experiment
scripts, while the reusable model/loss now live in the package.
"""

import torch

from climatesleuth.models import MonthlyRadiationTLWPNoPolesVGG, NTXentLoss


def main():
    # Recent no-poles monthly training used 9 channels: dynamic radiation/TLWP
    # variables plus static fields. Use input_shape to match your preprocessed grid. This sample uses a small dummy grid to run quickly.
    model = MonthlyRadiationTLWPNoPolesVGG(in_channels=9, projection_dim=128, masked_channel=4, input_shape=(32, 32))
    criterion = NTXentLoss(temperature=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Dummy paired views. Channel 4 is TLWP; zeros there are treated as masked cells.
    x_i = torch.randn(2, 9, 32, 32)
    x_j = torch.randn(2, 9, 32, 32)
    x_i[:, 4, :5, :] = 0.0
    x_j[:, 4, :5, :] = 0.0

    z_i = model(x_i)
    z_j = model(x_j)
    loss = criterion(z_i, z_j)
    loss.backward()
    optimizer.step()
    print(f"sample loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
