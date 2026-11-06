"""DAgger query scores: VIG and flow loss.

**VIG** (visual information gain). At one observation, draw S action chunks at full resolution and at each degraded
level (factors 2 ... 64), every level starting from the same S noises. Per degraded level: the 1-D Wasserstein
distance between the full-resolution and the degraded sample set, computed per chunk cell (16 steps x 8 action
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
    """Resolution sensitivity, optionally comparing per-sample temporal means.

    temporal_average=True reduces (levels, samples, time, action dimensions)
    over time before the Wasserstein comparison; samples remain distinct.
    """
    name = "vig"

    def __init__(self, num_samples: int = 32, temporal_average: bool = False, include_original: bool = False,
                 include_lowest_reference: bool = False, capture_samples: bool = False) -> None:
        self.num_samples = int(num_samples)
        self.temporal_average = bool(temporal_average)
        self.include_original = bool(include_original)
        self.include_lowest_reference = bool(include_lowest_reference)
        self.capture_samples = bool(capture_samples)

    @torch.inference_mode()
    def __call__(self, policies: PolicySet, state: torch.Tensor, images: torch.Tensor) -> Score:
        t0 = time.perf_counter()
        noise = torch.randn(self.num_samples, policies.horizon, policies.action_dim, device=policies.device)
        samples = policies.vig_samples(state, images, self.num_samples, noise).float()  # (F, S, H, D)
        # Average within each sampled chunk, retaining the sample distribution
        # and the eight action dimensions, before sorting for Wasserstein.
        flat = samples.mean(dim=2) if self.temporal_average else samples.reshape(len(policies.factors), self.num_samples, -1)
        gaps = {int(f): float(w1_per_cell(flat[0], flat[i])) for i, f in enumerate(policies.factors) if i > 0}
        value = max(gaps.values()) if gaps else 0.0
        details = {"w1_by_factor": gaps, "temporal_average": self.temporal_average}
        if self.include_original:
            original = samples.reshape(len(policies.factors), self.num_samples, -1)
            original_gaps = {int(f): float(w1_per_cell(original[0], original[i])) for i, f in enumerate(policies.factors) if i > 0}
            details.update(original_value=max(original_gaps.values()) if original_gaps else 0.,original_w1_by_factor=original_gaps)
        if self.include_lowest_reference:
            # Use the coarsest resolution as reference, retaining the original
            # per-action-cell aggregation, regardless of temporal_average.
            original = samples.reshape(len(policies.factors), self.num_samples, -1)
            reference = max(range(len(policies.factors)), key=lambda i: policies.factors[i])
            lowest_gaps = {int(f): float(w1_per_cell(original[reference], original[i]))
                           for i, f in enumerate(policies.factors) if i != reference}
            details.update(lowest_reference_factor=int(policies.factors[reference]),
                           lowest_reference_value=max(lowest_gaps.values()) if lowest_gaps else 0.,
                           lowest_reference_w1_by_factor=lowest_gaps)
        if self.capture_samples:
            details['samples'] = samples
        details['seconds'] = time.perf_counter() - t0
        return Score(value, samples[0], details)


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
