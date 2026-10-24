"""DAgger query scores: VIG and flow loss.

**VIG** (visual information gain). At one observation, draw S action chunks at full resolution and at each degraded
level (factors 2 ... 64), every level starting from the same S noises. Per degraded level: the 1-D Wasserstein
distance between the full-resolution and the degraded sample set, computed per chunk cell (16 steps x 7 action
dims, sorting each cell's S samples) and averaged over the cells. VIG is the largest of these over the degraded
levels. It is high where the plan depends on image detail. The samples come from one MoE policy routed per level,
or from one policy per level (``PolicySet``).

**Flow loss** (self-consistency). Draw one chunk from the full-resolution policy, then average the policy's own
flow-matching loss on that chunk over 512 random (noise, time) draws. It is high where the policy's field
disagrees with its own sample.

Both scores are computed on the full-resolution policy's normalized actions; both return the full-resolution
chunks the robot executes from (the first sample).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vig_dagger.policy import PolicySet


@dataclass
class Score:
    value: float
    chunks: torch.Tensor  # (S, horizon, action_dim) normalized chunks of the acting policy; chunks[0] is executed
    details: dict[str, Any] = field(default_factory=dict)


def w1_per_cell(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean over features of the 1-D Wasserstein distance between two sample sets x, y: (S, F)."""
    return (torch.sort(x, dim=0).values - torch.sort(y, dim=0).values).abs().mean()


class VIGScorer:
    name = "vig"

    def __init__(self, num_samples: int = 32) -> None:
        self.num_samples = int(num_samples)

    @torch.inference_mode()
    def __call__(self, policies: PolicySet, state: torch.Tensor, images: torch.Tensor) -> Score:
        t0 = time.perf_counter()
        noise = torch.randn(self.num_samples, policies.horizon, policies.action_dim, device=policies.device)
        samples = policies.vig_samples(state, images, self.num_samples, noise).float()  # (F, S, H, D)
        flat = samples.reshape(len(policies.factors), self.num_samples, -1)
        gaps = {int(f): float(w1_per_cell(flat[0], flat[i])) for i, f in enumerate(policies.factors) if i > 0}
        value = max(gaps.values()) if gaps else 0.0
        return Score(value, samples[0], {"w1_by_factor": gaps, "seconds": time.perf_counter() - t0})


class FlowLossScorer:
    name = "flow"

    def __init__(self, num_loss_samples: int = 512, num_action_samples: int = 1) -> None:
        self.num_loss_samples = int(num_loss_samples)
        self.num_action_samples = int(num_action_samples)

    @torch.inference_mode()
    def __call__(self, policies: PolicySet, state: torch.Tensor, images: torch.Tensor) -> Score:
        t0 = time.perf_counter()
        value, chunks = policies.flow_score(state, images, self.num_action_samples, self.num_loss_samples)
        return Score(value, chunks, {"seconds": time.perf_counter() - t0})


def build_scorer(name: str, settings: dict[str, Any]):
    """``vig`` or ``flow`` with the experiment config's ``scorers.<name>`` settings."""
    if name == "vig":
        return VIGScorer(num_samples=settings.get("num_samples", 32))
    if name == "flow":
        return FlowLossScorer(num_loss_samples=settings.get("num_loss_samples", 512),
                              num_action_samples=settings.get("num_action_samples", 1))
    raise ValueError(f"unknown scorer {name!r} (vig or flow)")
