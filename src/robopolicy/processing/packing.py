"""Canonical non-mutating input packing for training, validation, and inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from pydantic import model_validator
from torch import Tensor, nn
from typing_extensions import Self

from robopolicy.contracts.batch import PolicyInput
from robopolicy.contracts.schema import FeatureSchema, StrictConfig
from robopolicy.processing.input_modes import InputModesConfig
from robopolicy.processing.normalization import Normalizer, Statistics
from robopolicy.processing.resolution import ResolutionConfig, degrade_resolution

LEVEL_KEYS = ("resolution_indices", "mode_indices")  # per-sample categories packing appends as the last task slot


class ProcessingConfig(StrictConfig):
    """Tensor preprocessing settings; statistics fitting belongs to dataset readers."""

    image_size: tuple[int, int] | None = None
    crop_size: tuple[int, int] | None = None
    random_crop: bool = False
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    image_representation: Literal["rgb", "current_and_signed_difference"] = "rgb"
    resolution: ResolutionConfig | None = None
    input_modes: InputModesConfig | None = None

    @model_validator(mode="after")
    def validate_sizes(self) -> Self:
        """Reject invalid sizes, zero image scaling factors and two per-sample categories at once."""
        if self.resolution is not None and self.input_modes is not None:
            raise ValueError("Use either resolution levels or input modes; both would claim the last task slot")
        for size in (self.image_size, self.crop_size):
            if size is not None and min(size) < 1:
                raise ValueError("Image and crop sizes must be positive")
        if min(self.image_std) <= 0:
            raise ValueError("Image standard deviations must be positive")
        return self

    def validate_schema(self, schema: FeatureSchema) -> None:
        """Require images matching the configured representation and resolution levels."""
        images = next((feature for feature in schema.features if feature.key == "images"), None)
        if self.image_representation == "current_and_signed_difference":
            if images is None or images.source != "video" or images.history != 1 or images.future:
                raise ValueError("Signed differences require exactly two causal image observations")
        if self.resolution is not None:
            if images is None:
                raise ValueError("Resolution levels require image observations")
            height, width = images.shape[1:]
            if any(height % factor or width % factor for factor in self.resolution.factors):
                raise ValueError("Image height and width must be divisible by every factor")
        if self.input_modes is not None:
            if images is None:
                raise ValueError("Input modes require image observations")
            self.input_modes.camera_mask(images.cameras)  # rejects unknown camera names

    @property
    def levels(self) -> tuple[str, int] | None:
        """The per-sample category packing appends (its batch key, its number of values), if any."""
        if self.resolution is not None:
            return "resolution_indices", len(self.resolution.factors)
        if self.input_modes is not None:
            return "mode_indices", len(self.input_modes.hidden)
        return None


class InputPacker(nn.Module):
    """Convert raw batches to the common PolicyInput using supplied fixed statistics."""

    def __init__(
        self, schema: FeatureSchema, statistics: Statistics, config: ProcessingConfig | None = None
    ) -> None:
        """Bind the saved schema and statistics without consulting any dataset."""
        super().__init__()
        if schema != statistics.feature_schema:
            raise ValueError("Input schema and normalization schema differ")
        self.schema = schema
        self.normalizer = Normalizer(statistics)
        self.config = config or ProcessingConfig()
        self.config.validate_schema(schema)
        self._image_mean: Tensor
        self._image_std: Tensor
        self.register_buffer(
            "_image_mean",
            torch.tensor(self.config.image_mean).reshape(1, 3, 1, 1),
            persistent=False,
        )  # (1, 3, 1, 1)
        self.register_buffer(
            "_image_std",
            torch.tensor(self.config.image_std).reshape(1, 3, 1, 1),
            persistent=False,
        )  # (1, 3, 1, 1)
        self._mode_camera_mask: Tensor | None
        mode_mask = None
        if self.config.input_modes is not None:
            images = next(feature for feature in schema.features if feature.key == "images")
            mode_mask = self.config.input_modes.camera_mask(images.cameras)  # (M, N_camera) bool, True = hidden
        self.register_buffer("_mode_camera_mask", mode_mask, persistent=False)

    def images(self, images: Tensor, training: bool) -> Tensor:
        """Preprocess uint8 ``(B, T, N, 3, H, W)`` without changing caller storage.

        Difference mode replaces the previous image with a signed, gray-centered
        difference at unit gain. Geometry is shared and normalization happens last.
        """
        if images.dtype != torch.uint8 or images.ndim != 6:
            raise ValueError("Raw images must be RGB uint8 (B, T_obs, N_camera, 3, H, W)")
        batch, steps, cameras = images.shape[:3]
        if self.config.image_representation == "current_and_signed_difference" and steps != 2:
            raise ValueError("Signed differences require exactly two causal image observations")
        pixels = images.flatten(0, 2).to(torch.float32) / 255.0
        # Shape: (B*T_obs*N_camera, 3, H, W), now owned floating pixels.
        if self.config.crop_size is not None:
            height, width = pixels.shape[-2:]
            crop_h, crop_w = self.config.crop_size
            if crop_h > height or crop_w > width:
                raise ValueError("Crop size exceeds supplied image dimensions")
            if training and self.config.random_crop:
                top = int(torch.randint(height - crop_h + 1, ()).item())
                left = int(torch.randint(width - crop_w + 1, ()).item())
            else:
                top, left = (height - crop_h) // 2, (width - crop_w) // 2
            pixels = pixels[..., top : top + crop_h, left : left + crop_w]
            # Shape: (B*T_obs*N_camera, 3, crop_h, crop_w).
        if self.config.image_size is not None:
            pixels = functional.interpolate(
                pixels,
                size=self.config.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # Shape: (B*T_obs*N_camera, 3, resized_h, resized_w).
        if self.config.image_representation == "current_and_signed_difference":
            frames = pixels.reshape(batch, steps, cameras, *pixels.shape[1:])
            # Shape: (B, 2, N_camera, 3, output_h, output_w).
            difference = 0.5 + 0.5 * (frames[:, 1] - frames[:, 0])
            pixels = torch.stack((difference, frames[:, 1]), dim=1).flatten(0, 2)
            # Shape: (B*2*N_camera, 3, output_h, output_w), ordered [difference, current].
        mean = self._image_mean.to(pixels)
        std = self._image_std.to(pixels)
        pixels = (pixels - mean) / std
        return pixels.reshape(batch, steps, cameras, *pixels.shape[1:])
        # Shape: (B, T_obs, N_camera, 3, output_h, output_w).

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        device: str = "cpu",
        training: bool = False,
        require_actions: bool = False,
    ) -> PolicyInput:
        """Pack supplied observations, adding targets only when they are present.

        Args:
            batch: Raw tensors in declared temporal shapes and optional metadata.
                Resolution policies accept int64 ``resolution_indices`` ``(B, 1)``.
            device: Destination for every tensor, preserving categorical integers.
            training: Enable configured random cropping and resolution sampling.
            require_actions: Fail clearly if training targets are absent.

        Returns:
            A new PolicyInput; no caller-owned tensor or array is modified.
        """
        values = self.validate(
            batch,
            require_actions=require_actions,
            require_resolution=not (training or require_actions),
        )
        moved = {key: value.to(device, non_blocking=True) for key, value in values.items()}
        return self.pack_tensors(moved, training=training)

    def validate(
        self,
        batch: Mapping[str, Any],
        *,
        require_actions: bool = False,
        require_resolution: bool | None = None,
    ) -> dict[str, Tensor]:
        """Validate raw feature shapes/dtypes and return tensor references.

        Args:
            batch: Raw tensors with batch and declared temporal dimensions.
            require_actions: Require action targets for training.
            require_resolution: Require ``resolution_indices`` for resolution
                policies. Defaults to inference batches (``not require_actions``);
                training and validation may omit them and have levels assigned.

        Returns:
            Declared tensors in schema order, without copies or device transfers,
            followed by any supplied ``resolution_indices``.
        """
        values: dict[str, Tensor] = {}
        for feature in self.schema.features:
            key = feature.key
            if key not in batch:
                if key == "actions" and not require_actions:
                    continue
                raise ValueError(f"Missing required input feature: {key}")
            value = batch[key]
            if not isinstance(value, Tensor):
                raise ValueError(f"Input feature {key} must be a Tensor")
            expected = (
                (feature.window_size, len(feature.cameras), *feature.shape)
                if feature.source == "video"
                else (feature.window_size, *feature.shape)
            )
            if tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"Input shape mismatch for {key}: {tuple(value.shape[1:])} != {expected}"
                )
            if value.dtype != getattr(torch, feature.dtype):
                raise ValueError(
                    f"Input dtype mismatch for {key}: expected {feature.dtype}, got {value.dtype}"
                )
            if key == "images" and self.config.crop_size is not None:
                if any(crop > size for crop, size in zip(self.config.crop_size, value.shape[-2:])):
                    raise ValueError("Crop size exceeds supplied image dimensions")
            values[key] = value
        if "state" not in values or (require_actions and "actions" not in values):
            raise ValueError("Packing requires state and, for training, actions")
        allowed = {"state", "actions", "images", "entities", "task_indices", "entity_indices"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"Feature keys need a project adapter before packing: {sorted(unknown)}"
            )
        if self.config.levels is not None:
            key, levels = self.config.levels
            indices = batch.get(key)
            if require_resolution is None:
                require_resolution = not require_actions
            if indices is None:
                if require_resolution:
                    kind = "Resolution" if key == "resolution_indices" else "Input-mode"
                    raise ValueError(f"{kind} policies require int64 {key} shaped (B, 1)")
            else:
                if (
                    not isinstance(indices, Tensor)
                    or indices.dtype != torch.int64
                    or indices.ndim != 2
                    or indices.shape[1] != 1
                ):
                    raise ValueError(f"{key} must be int64 shaped (B, 1)")
                if bool(((indices < 0) | (indices >= levels)).any()):
                    raise ValueError(f"{key} must be in [0, {levels})")
                values[key] = indices
        batch_size = values["state"].shape[0]
        if batch_size < 1 or any(value.shape[0] != batch_size for value in values.values()):
            raise ValueError("Input features must share a nonempty batch dimension")
        return values

    @staticmethod
    def canonical_inputs(values: Mapping[str, Tensor]) -> PolicyInput:
        """Wrap tensors and remove the current-step axis from categorical inputs.

        Args:
            values: Features with shapes validated against their schema, optionally
                with ``resolution_indices`` or ``mode_indices`` ``(B, 1)`` and a
                ``camera_mask`` ``(B, N_camera)``.

        Returns:
            Policy inputs; categorical ``(B, 1, K)`` tensors become ``(B, K)``. A
            resolution or mode index becomes the last ``task_indices`` slot.
        """
        tensors = dict(values)
        camera_mask = tensors.pop("camera_mask", None)
        level = next((tensors.pop(key) for key in LEVEL_KEYS if key in tensors), None)
        if level is not None:
            slots = [level[:, :, None]]  # Shape: (B, 1, 1)
            if "task_indices" in tensors:
                slots.insert(0, tensors["task_indices"])
            tensors["task_indices"] = torch.cat(slots, dim=-1)  # Shape: (B, 1, K)
        return PolicyInput(
            **{
                key: tensor[:, 0] if key in {"task_indices", "entity_indices"} else tensor
                for key, tensor in tensors.items()
            },
            camera_mask=camera_mask,
        )

    def pack_tensors(self, values: Mapping[str, Tensor], *, training: bool = False) -> PolicyInput:
        """Preprocess validated tensors already placed on the destination device.

        Args:
            values: Raw tensors in their declared temporal shapes.
            training: Enable configured random cropping for training only.

        Returns:
            Canonical inputs, including normalized floating images and features.
        """
        values = dict(values)
        if self.config.input_modes is not None:
            assert self._mode_camera_mask is not None
            values["mode_indices"] = self.mode_indices(values, training)
            values["camera_mask"] = self._mode_camera_mask.to(values["state"].device)[values["mode_indices"][:, 0]]
            # Shape: (B, N_camera) bool, True = hidden from the action head.
        if self.config.resolution is not None:
            values["resolution_indices"] = self.resolution_indices(values, training)
            values["images"] = degrade_resolution(
                values["images"],
                values["resolution_indices"][:, 0],
                self.config.resolution.factors,
            )
        transformed = {
            key: self.images(tensor, training)
            if key == "images"
            else tensor
            if key in (*LEVEL_KEYS, "camera_mask")
            else self.normalizer.transform(key, tensor)
            for key, tensor in values.items()
        }
        return self.canonical_inputs(transformed)

    def resolution_indices(self, values: Mapping[str, Tensor], training: bool) -> Tensor:
        """Return supplied resolution levels or draw them for a batch without levels.

        Training draws a uniform level per sample. Validation without levels uses
        the deterministic cycle ``0, 1, ..., L-1, 0, ...`` across the batch, so its
        loss covers every level reproducibly.

        Args:
            values: Validated raw tensors, optionally with ``resolution_indices``.
            training: Draw random levels rather than the validation cycle.

        Returns:
            Integer levels ``(B, 1)`` on the state device.
        """
        assert self.config.resolution is not None
        return self._levels(values.get("resolution_indices"), values["state"], len(self.config.resolution.factors), training)

    def mode_indices(self, values: Mapping[str, Tensor], training: bool) -> Tensor:
        """Return supplied input modes or draw them like resolution levels (see ``resolution_indices``)."""
        assert self.config.input_modes is not None
        return self._levels(values.get("mode_indices"), values["state"], len(self.config.input_modes.hidden), training)

    @staticmethod
    def _levels(supplied: Tensor | None, state: Tensor, levels: int, training: bool) -> Tensor:
        """Supplied categories, a uniform draw per sample in training, else the cycle 0..L-1 (``(B, 1)``)."""
        if supplied is not None:
            return supplied
        if training:
            return torch.randint(levels, (state.shape[0], 1), device=state.device)
        return (torch.arange(state.shape[0], device=state.device) % levels)[:, None]

    def actions_to_raw(self, actions: Tensor) -> Tensor:
        """Invert action normalization for predictions ``(B, T_action, D_action)``."""
        return self.normalizer.transform("actions", actions, inverse=True)
