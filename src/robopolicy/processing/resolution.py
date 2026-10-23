"""Image-resolution degradation whose level is supplied to the policy as a category.

A degraded frame is area-downsampled by an integer factor and bilinearly restored
to its original size, with both stages rounded to uint8 like an OpenCV
``INTER_AREA``/``INTER_LINEAR`` round trip on stored frames. Applying it inside
input packing gives training, validation and inference one implementation, and the
factor's index joins ``task_indices`` as the last categorical slot.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from pydantic import Field, field_validator
from torch import Tensor

from robopolicy.contracts.schema import StrictConfig


class ResolutionConfig(StrictConfig):
    """Configure resolution levels; the index of a factor is its category.

    Attributes:
        factors: Integer downsampling factors; ``1`` keeps the original frame.
    """

    factors: tuple[int, ...] = Field(default=(1, 2, 4, 8), min_length=1)

    @field_validator("factors")
    @classmethod
    def validate_factors(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        """Reject non-positive or repeated factors, whose categories would be ambiguous."""
        if any(value < 1 for value in values) or len(set(values)) != len(values):
            raise ValueError("Resolution factors must be unique positive integers")
        return values


def degrade_resolution(images: Tensor, factor_indices: Tensor, factors: tuple[int, ...]) -> Tensor:
    """Degrade every sample's frames by its own factor with static tensor shapes.

    Every configured factor is computed for the whole batch and selected per sample,
    so the computation has no data-dependent shapes or host synchronization.

    Args:
        images (Tensor): Raw RGB uint8 images. Shape: ``(B, T, N, 3, H, W)``; ``H``
            and ``W`` must be divisible by every factor.
        factor_indices (Tensor): Integer index into ``factors`` per sample.
            Shape: ``(B,)``.
        factors (tuple[int, ...]): Downsampling factors addressed by the indices.

    Returns:
        Tensor: Degraded uint8 images. Shape: ``(B, T, N, 3, H, W)``.
    """
    batch, steps, cameras, channels, height, width = images.shape
    pixels = images.flatten(0, 2).to(torch.float32)  # Shape: (B*T*N, 3, H, W) in [0, 255]
    frame_indices = factor_indices.repeat_interleave(steps * cameras)  # Shape: (B*T*N,)
    degraded = pixels
    for index, factor in enumerate(factors):
        if factor == 1:
            continue
        # Integer-factor area resampling is a block mean; the converter stores the
        # reduced frame as uint8 before restoring it.
        reduced = functional.avg_pool2d(pixels, kernel_size=factor).round()
        # Shape: (B*T*N, 3, H/factor, W/factor).
        restored = functional.interpolate(
            reduced, size=(height, width), mode="bilinear", align_corners=False
        )  # Shape: (B*T*N, 3, H, W); half-pixel centers like OpenCV INTER_LINEAR.
        selected = (frame_indices == index)[:, None, None, None]  # Shape: (B*T*N, 1, 1, 1)
        degraded = torch.where(selected, restored.round().clamp(0, 255), degraded)
    return degraded.to(torch.uint8).reshape(batch, steps, cameras, channels, height, width)
