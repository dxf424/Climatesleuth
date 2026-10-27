"""Monthly radiation/TLWP contrastive VGG models.

The model in this module is the reusable package version of the architecture
used by ``training_monthly_radiation_TLWP_no_poles.py``. It is designed for
monthly no-poles fields containing radiation channels, TLWP, and optional static
channels. The first convolution treats the TLWP channel specially: zero-valued
TLWP cells are interpreted as invalid/masked and the convolution is renormalized
by the number of valid cells in each local window.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAwareMaskedConv2d(nn.Module):
    """Convolution with a locally renormalized masked input channel.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of convolution filters.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        bias: Whether to use a bias term.
        masked_channel: Index of the channel that contains masked values.
        eps: Small value used to avoid division by zero.

    Notes:
        Only ``masked_channel`` is multiplied by ``mask`` and renormalized. All
        other channels pass through the same convolution weights normally. This
        is useful for TLWP fields where zeros mark no-data/land/pole-masked
        regions, but the other dynamic/static channels should still be available.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        bias: bool = True,
        masked_channel: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0 <= masked_channel < in_channels:
            raise ValueError("masked_channel must be a valid input-channel index")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.masked_channel = masked_channel
        self.eps = eps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kernel_h = kh
        self.kernel_w = kw
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.register_buffer("ones_kernel", torch.ones(1, 1, kh, kw))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply the channel-aware masked convolution.

        Args:
            x: Input tensor with shape ``(batch, channels, lat, lon)``.
            mask: Binary/float mask with shape ``(batch, 1, lat, lon)`` or
                ``(batch, lat, lon)``. A single-batch mask is broadcast over the
                input batch.
        """
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, lat, lon)")
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError("mask must have shape (batch, 1, lat, lon) or (batch, lat, lon)")

        mask = mask.to(dtype=x.dtype, device=x.device)
        if mask.shape[0] == 1 and x.shape[0] > 1:
            mask = mask.expand(x.shape[0], -1, -1, -1)

        x_masked_channel = x[:, self.masked_channel : self.masked_channel + 1] * mask
        other_idx = [i for i in range(self.in_channels) if i != self.masked_channel]
        x_other = x[:, other_idx]

        weight = self.conv.weight
        out_other = F.conv2d(x_other, weight[:, other_idx], bias=None, stride=self.stride, padding=self.padding)
        out_masked = F.conv2d(
            x_masked_channel,
            weight[:, self.masked_channel : self.masked_channel + 1],
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )
        valid_count = F.conv2d(mask, self.ones_kernel, bias=None, stride=self.stride, padding=self.padding)
        scale = float(self.kernel_h * self.kernel_w) / (valid_count + self.eps)
        out_masked = torch.where(valid_count > 0, out_masked * scale, torch.zeros_like(out_masked))
        out = out_other + out_masked
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1, -1, 1, 1)
        return out


class MonthlyRadiationTLWPNoPolesVGG(nn.Module):
    """VGG-style contrastive encoder for monthly radiation/TLWP no-poles data.

    Args:
        in_channels: Number of input channels after dynamic and static fields
            have been concatenated. The recent monthly workflow uses ``9``.
        projection_dim: Size of the contrastive embedding.
        masked_channel: Index of the TLWP channel used to infer the water mask.
        input_shape: Spatial shape ``(lat, lon)`` after removing poles. Used to
            infer the flattened projector size.
        warn_zero_frac_low: Warn during training if the TLWP channel contains
            too few zeros to plausibly encode a mask.
        warn_zero_frac_high: Warn during training if the TLWP channel is almost
            entirely zero.
        check_zero_fraction: Whether to emit the above sanity warnings.

    Forward input:
        Tensor with shape ``(batch, in_channels, lat, lon)``.

    Returns:
        Tensor with shape ``(batch, projection_dim)``.
    """

    def __init__(
        self,
        in_channels: int = 9,
        projection_dim: int = 128,
        masked_channel: int = 4,
        input_shape=(128, 288),
        warn_zero_frac_low: float = 0.001,
        warn_zero_frac_high: float = 0.95,
        check_zero_fraction: bool = True,
    ) -> None:
        super().__init__()
        self.masked_channel = masked_channel
        self.warn_zero_frac_low = warn_zero_frac_low
        self.warn_zero_frac_high = warn_zero_frac_high
        self.check_zero_fraction = check_zero_fraction

        self.conv1 = ChannelAwareMaskedConv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,
            masked_channel=masked_channel,
            bias=True,
        )
        self.backbone_rest = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )
        flat_features = self._infer_flat_features(in_channels, input_shape)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
            nn.GELU(),
            nn.Linear(256, projection_dim),
        )

    def _infer_flat_features(self, in_channels: int, input_shape) -> int:
        with torch.no_grad():
            dummy = torch.ones(1, in_channels, input_shape[0], input_shape[1])
            mask = torch.ones(1, 1, input_shape[0], input_shape[1])
            y = self.backbone_rest(self.conv1(dummy, mask))
        return int(y.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, lat, lon)")
        tlwp = x[:, self.masked_channel : self.masked_channel + 1]
        water_mask = (tlwp != 0).to(dtype=x.dtype)

        if self.check_zero_fraction and self.training:
            zero_frac = (tlwp == 0).float().mean().item()
            if zero_frac < self.warn_zero_frac_low:
                warnings.warn(
                    f"[MonthlyRadiationTLWPNoPolesVGG] Channel {self.masked_channel} has very few zeros "
                    f"(zero fraction={zero_frac:.6f}); verify masked_channel.",
                    stacklevel=2,
                )
            elif zero_frac > self.warn_zero_frac_high:
                warnings.warn(
                    f"[MonthlyRadiationTLWPNoPolesVGG] Channel {self.masked_channel} is mostly zeros "
                    f"(zero fraction={zero_frac:.6f}); verify preprocessing.",
                    stacklevel=2,
                )

        x = x.clone()
        x[:, self.masked_channel : self.masked_channel + 1] = tlwp * water_mask
        x = self.conv1(x, water_mask)
        x = self.backbone_rest(x)
        return self.projector(x)


# Backward-compatible short alias used in experiment scripts.
MeanPooledMonthlyRadiationTLWPVGG = MonthlyRadiationTLWPNoPolesVGG
