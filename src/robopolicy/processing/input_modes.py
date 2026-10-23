"""Input modes: per-sample hiding of camera groups, whose index is supplied to the policy as a category.

A mode lists the cameras it hides. Hidden cameras are still encoded, but their tokens are excluded from the action
head's cross-attention by a key padding mask, so they cannot influence the actions; the head's other conditioning
(time and robot state) never sees image tokens. Packing draws a mode per training sample, cycles the modes in
validation, and requires ``mode_indices`` ``(B, 1)`` at inference; the mode's index joins ``task_indices`` as the
last categorical slot, so a sparse-expert head can route by it.
"""

from __future__ import annotations

import torch
from pydantic import Field, model_validator
from torch import Tensor
from typing_extensions import Self

from robopolicy.contracts.schema import StrictConfig


class InputModesConfig(StrictConfig):
    """Configure input modes; the position of a mode is its category.

    Attributes:
        hidden: For every mode, the camera names it hides (empty: every camera is visible).
        names: Optional display labels, one per mode.
    """

    hidden: tuple[tuple[str, ...], ...] = Field(min_length=2)
    names: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_modes(self) -> Self:
        """Reject repeated cameras within a mode, repeated modes and mislabeled modes."""
        for cameras in self.hidden:
            if len(set(cameras)) != len(cameras):
                raise ValueError("A mode lists a camera more than once")
        if len({frozenset(cameras) for cameras in self.hidden}) != len(self.hidden):
            raise ValueError("Input modes must hide different camera sets")
        if self.names is not None and len(self.names) != len(self.hidden):
            raise ValueError("Give one name per input mode")
        return self

    def camera_mask(self, cameras: tuple[str, ...]) -> Tensor:
        """Hidden cameras per mode as bool ``(M, N_camera)`` in the schema's camera order."""
        unknown = sorted({name for mode in self.hidden for name in mode} - set(cameras))
        if unknown:
            raise ValueError(f"Input modes hide unknown cameras {unknown}; the images declare {list(cameras)}")
        return torch.tensor([[camera in mode for camera in cameras] for mode in self.hidden], dtype=torch.bool)
