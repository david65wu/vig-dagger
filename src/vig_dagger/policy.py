"""Running trained policies in the simulator: one checkpoint (`LibraryPolicy`) and what DAgger drives (`PolicySet`).

Policy inputs are the raw robot state (7 arm joint positions + 2 finger positions) and the undegraded RGB renders of
the policy's cameras. A policy degrades its images itself (``processing.resolution``): the 7-level MoE routes each
sample by its level, and a per-resolution policy always applies its one factor. Actions are absolute joint-position
targets (7 joints + gripper), predicted as 16-step chunks of which DAgger and evaluation execute the first 8.

VIG compares normalized action samples. With one MoE policy every level shares the policy's action normalization;
with one policy per level each policy's samples are mapped into the full-resolution policy's normalization
(denormalized with their own statistics, renormalized with the full-resolution policy's).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch

from vig_dagger.envs import POLICY_CAMERAS


def robot_state(obs: dict[str, Any], device: torch.device) -> torch.Tensor:
    """(1, 9): the 7 arm joint positions and the 2 finger positions, as in the training data."""
    qpos = obs["agent"]["qpos"].to(device=device, dtype=torch.float32)
    return torch.cat([qpos[:, :7], qpos[:, -2:]], dim=-1)


def camera_images(obs: dict[str, Any], cameras: list[str] | tuple[str, ...], device: torch.device) -> torch.Tensor:
    """(1, n_cams, 3, H, W) undegraded RGB uint8 renders in the policy's camera order."""
    return torch.stack([obs["sensor_data"][name]["rgb"].permute(0, 3, 1, 2).to(device=device) for name in cameras], dim=1)


class LibraryPolicy:
    """Eager inference for one robopolicy checkpoint, with optional resolution routing.

    Attributes:
        cameras: Camera names in the order the policy expects.
        horizon, action_dim: Chunk length (16) and action width (8).
        factors: Downsample factors whose index is the policy's resolution level, or (1,) without levels.
    """

    def __init__(self, checkpoint: str | Path, device: torch.device | str) -> None:
        from robopolicy import InferenceConfig, load_policy

        self.device = torch.device(device)
        # Eager execution: VIG batches change size, which a compiled runtime would reject.
        self.runtime = load_policy(checkpoint, device=str(self.device), inference_config=InferenceConfig(compile="disabled"))
        self.policy, self.packer = self.runtime.policy, self.runtime.packer
        config = self.policy.config
        if config.obs_steps != 1:
            raise ValueError(f"expects one observation step, got {config.obs_steps}")
        self.cameras = list(self.packer.schema["images"].cameras)
        self.horizon, self.action_dim = config.action_steps, config.action_dim
        resolution = self.packer.config.resolution
        self.factors = tuple(resolution.factors) if resolution is not None else (1,)
        self.routed = resolution is not None
        self.checkpoint = str(checkpoint)

    def level(self, factor: int) -> int:
        if int(factor) not in self.factors:
            raise ValueError(f"factor {factor} is not one of the policy's levels {self.factors}")
        return self.factors.index(int(factor))

    def _batch(self, state: torch.Tensor, images: torch.Tensor, factors: list[int]) -> dict[str, torch.Tensor]:
        """One raw observation (state (1, 9), images (1, n_cams, 3, H, W)) repeated per factor."""
        count = len(factors)
        batch = {
            "state": state.reshape(1, 1, -1).to(device=self.device, dtype=torch.float32).repeat(count, 1, 1),
            "images": images.reshape(1, 1, *images.shape[-4:]).to(self.device).repeat(count, 1, 1, 1, 1, 1),
        }
        if self.routed:
            batch["resolution_indices"] = torch.tensor([[self.level(f)] for f in factors], dtype=torch.int64, device=self.device)
        elif any(int(f) != 1 for f in factors):
            raise ValueError("this policy has no resolution levels; only factor 1 is available")
        return batch

    def _field(self, inputs, repeats: int) -> partial:
        """The action field with each sample's router inputs bound, repeated to match a repeat_interleave'd context."""
        kwargs = {k: v.repeat_interleave(repeats, dim=0) for k, v in self.policy.field_kwargs(inputs).items()}
        return partial(self.policy.field, **kwargs)

    @torch.inference_mode()
    def act(self, state: torch.Tensor, images: torch.Tensor, factor: int | None = None) -> torch.Tensor:
        """Raw joint-target chunk (1, horizon, action_dim) at one level (default: the policy's first)."""
        return self.runtime.predict(self._batch(state, images, [self.factors[0] if factor is None else int(factor)]))

    @torch.inference_mode()
    def sample_normalized(self, state: torch.Tensor, images: torch.Tensor, factors: list[int] | tuple[int, ...],
                          num_samples: int, noise: torch.Tensor) -> torch.Tensor:
        """Normalized action samples (F, S, horizon, action_dim), one block per factor, from noise (F*S, H, D).

        Each level is encoded once and its context repeated over the S samples.
        """
        factors = [int(f) for f in factors]
        count, samples = len(factors), int(num_samples)
        inputs = self.packer(self._batch(state, images, factors), device=str(self.device))
        context, _ = self.policy.encode(inputs)  # (F, L, D)
        actions = self.policy.algorithm.generate_samples(
            self._field(inputs, samples), noise.to(context.dtype), context.repeat_interleave(samples, dim=0))
        return actions.reshape(count, samples, self.horizon, self.action_dim)

    @torch.inference_mode()
    def flow_score(self, state: torch.Tensor, images: torch.Tensor, num_action_samples: int = 1,
                   num_loss_samples: int = 512, factor: int | None = None) -> tuple[float, torch.Tensor]:
        """Flow-loss self-consistency: draw num_action_samples chunks from fresh noise, then the mean flow-matching loss
        of the first chunk over num_loss_samples random (noise, time) draws.

        Returns:
            The loss and the normalized chunks (num_action_samples, horizon, action_dim) to execute from.
        """
        level = self.factors[0] if factor is None else int(factor)
        inputs = self.packer(self._batch(state, images, [level]), device=str(self.device))
        context, _ = self.policy.encode(inputs)  # (1, L, D)
        count, draws = int(num_action_samples), int(num_loss_samples)
        noise = torch.randn(count, self.horizon, self.action_dim, device=self.device, dtype=context.dtype)
        chunks = self.policy.algorithm.generate_samples(self._field(inputs, count), noise, context.repeat_interleave(count, dim=0))
        loss = self.policy.algorithm.compute_loss(
            self._field(inputs, draws), chunks[0:1].repeat(draws, 1, 1), context.repeat_interleave(draws, dim=0))
        return float(loss), chunks

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalized (..., horizon, action_dim) actions to raw joint targets."""
        shape = actions.shape
        return self.packer.actions_to_raw(actions.reshape(-1, *shape[-2:])).reshape(shape)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw joint targets (..., action_dim) into this policy's action normalization."""
        return self.packer.normalizer.transform("actions", actions)


class PolicySet:
    """The policy DAgger and evaluation drive, plus where VIG's degraded-level samples come from.

    ``moe``: one checkpoint with resolution levels; it acts at full resolution and VIG samples it at every level.
    ``per_resolution``: one checkpoint per factor; the full-resolution one acts and VIG samples each level from its own
    policy, all starting from the same noise.

    Attributes:
        factors: Levels VIG compares, full resolution (1) first.
        cameras, horizon, action_dim: From the acting policy.
    """

    def __init__(self, policies: dict[int, LibraryPolicy]) -> None:
        if 1 not in policies:
            raise ValueError("a policy set needs a full-resolution (factor 1) policy")
        self.policies = dict(sorted(policies.items()))
        self.actor = self.policies[1]
        if len(self.policies) == 1:  # one routed policy covers every level
            self.factors = self.actor.factors
            if self.factors[0] != 1:
                raise ValueError(f"the policy's first level must be full resolution, got {self.factors}")
        else:
            self.factors = tuple(self.policies)
            for factor, policy in self.policies.items():
                if policy.factors != (factor,):
                    raise ValueError(f"the factor-{factor} policy has levels {policy.factors}, expected ({factor},)")
                if (policy.horizon, policy.action_dim, policy.cameras) != (self.actor.horizon, self.actor.action_dim, self.actor.cameras):
                    raise ValueError(f"the factor-{factor} policy has a different chunk shape or camera order")
        self.cameras, self.horizon, self.action_dim = self.actor.cameras, self.actor.horizon, self.actor.action_dim
        self.device = self.actor.device

    @classmethod
    def load(cls, checkpoints: str | Path | dict[int, str | Path], device: torch.device | str) -> PolicySet:
        """One MoE checkpoint (a path), or a {factor: path} mapping of per-resolution checkpoints."""
        if isinstance(checkpoints, (str, Path)):
            return cls({1: LibraryPolicy(checkpoints, device)})
        return cls({int(f): LibraryPolicy(path, device) for f, path in checkpoints.items()})

    def act(self, state: torch.Tensor, images: torch.Tensor, factor: int = 1) -> torch.Tensor:
        """Raw chunk (1, horizon, action_dim) from the acting policy (a MoE policy can also act at another level)."""
        return self.actor.act(state, images, factor)

    @torch.inference_mode()
    def vig_samples(self, state: torch.Tensor, images: torch.Tensor, num_samples: int, noise: torch.Tensor) -> torch.Tensor:
        """Normalized samples (F, S, horizon, action_dim) per level in self.factors, in the acting policy's
        normalization, every level from the same noise (S, horizon, action_dim)."""
        count = len(self.factors)
        if len(self.policies) == 1:
            return self.actor.sample_normalized(state, images, self.factors, num_samples, noise.repeat(count, 1, 1))
        blocks = []
        for factor, policy in self.policies.items():
            samples = policy.sample_normalized(state, images, [factor], num_samples, noise)[0]
            if policy is not self.actor:
                samples = self.actor.normalize_actions(policy.denormalize_actions(samples))
            blocks.append(samples)
        return torch.stack(blocks)

    def flow_score(self, state: torch.Tensor, images: torch.Tensor, num_action_samples: int = 1,
                   num_loss_samples: int = 512) -> tuple[float, torch.Tensor]:
        """The acting policy's flow loss at full resolution, and its normalized chunks."""
        return self.actor.flow_score(state, images, num_action_samples, num_loss_samples)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.actor.denormalize_actions(actions)


def check_cameras(policies: PolicySet) -> None:
    if tuple(policies.cameras) != POLICY_CAMERAS:
        raise ValueError(f"policy cameras {policies.cameras} differ from the task's {POLICY_CAMERAS}")
