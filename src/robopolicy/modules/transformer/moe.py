"""Sparse mixture-of-experts feed-forward layers with task-specific routers.

This follows Sparse Diffusion Policy (Wang et al., CoRL 2024, arXiv:2407.01531) and
its released implementation: each Transformer feed-forward block becomes a set of
smaller expert MLPs, and one router per task selects the top-k experts for every
token. Gates are the router's softmax probabilities of the selected experts without
renormalization. Instead of a load-balancing loss, a mutual-information objective
encourages each task to concentrate on its own experts while experts remain shared.

Every expert is evaluated densely and unselected outputs receive an exact zero gate.
Outputs and gradients therefore equal sparse dispatch, while shapes stay static for
``torch.compile``, CUDA graphs, AMP and mixed-task batches without host
synchronization. This favors the small action heads used here over FLOP savings.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

# SDP's floor inside logarithms of routing-probability estimates.
_LOG_EPSILON = 1e-4


class TaskRouters(nn.Module):
    """Score experts for each token with one two-layer MLP per task router.

    SDP selects a router by task identity and gives it only the token features. The
    weights of all routers are stacked so each sample gathers its own router, which
    lets one batch mix tasks.

    Attributes:
        num_routers (int): Number of independent task routers ``R``.
        num_experts (int): Number of experts ``E`` scored by every router.
        hidden_weight (nn.Parameter): First-layer weights, shape ``(R, D, D_router)``.
        hidden_bias (nn.Parameter): First-layer biases, shape ``(R, D_router)``.
        output_weight (nn.Parameter): Expert-logit weights, shape ``(R, D_router, E)``.
        output_bias (nn.Parameter): Expert-logit biases, shape ``(R, E)``.
    """

    def __init__(self, dim: int, num_experts: int, num_routers: int) -> None:
        """Allocate stacked router MLPs with SDP's ``D -> D/4 -> E`` widths.

        Args:
            dim: Token feature dimension ``D``.
            num_experts: Number of experts ``E`` each router scores.
            num_routers: Number of task routers ``R``.
        """
        super().__init__()
        hidden_dim = max(1, dim // 4)
        self.num_routers = num_routers
        self.num_experts = num_experts
        self.hidden_weight = nn.Parameter(torch.empty(num_routers, dim, hidden_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(num_routers, hidden_dim))
        self.output_weight = nn.Parameter(torch.empty(num_routers, hidden_dim, num_experts))
        self.output_bias = nn.Parameter(torch.zeros(num_routers, num_experts))
        # SDP's trained models used N(0, 0.02) router weights with zero biases. The
        # small random logits give near-uniform probabilities whose ties break per
        # token; an exact zero initialization would send every token to the same
        # first experts.
        nn.init.normal_(self.hidden_weight, std=0.02)
        nn.init.normal_(self.output_weight, std=0.02)

    def forward(self, tokens: Tensor, router_indices: Tensor) -> Tensor:
        """Compute expert logits using each sample's own task router.

        Args:
            tokens (Tensor): Token features. Shape: ``(B, T, D)``.
            router_indices (Tensor): Integer router of every sample. Shape: ``(B,)``.

        Returns:
            Tensor: Expert logits. Shape: ``(B, T, E)``.
        """
        hidden_weight = self.hidden_weight[router_indices]  # Shape: (B, D, D_router)
        hidden_bias = self.hidden_bias[router_indices][:, None]  # Shape: (B, 1, D_router)
        hidden = functional.gelu(torch.einsum("btd,bdh->bth", tokens, hidden_weight) + hidden_bias)
        # Shape: (B, T, D_router).
        output_weight = self.output_weight[router_indices]  # Shape: (B, D_router, E)
        output_bias = self.output_bias[router_indices][:, None]  # Shape: (B, 1, E)
        return torch.einsum("bth,bhe->bte", hidden, output_weight) + output_bias
        # Shape: (B, T, E).


class TaskExpertMutualInformation(nn.Module):
    """Track ``p(task, expert)`` and return SDP's mutual-information surrogate.

    The joint distribution is the routing probability mass per task, averaged over
    tokens, accumulated as a running mean during warmup and then as an exponential
    moving average. The returned loss is the first-order surrogate of ``-I(T; E)``:
    its gradient with respect to the current batch's joint estimate equals the
    negative gradient of the mutual information evaluated at the stored estimate.
    Including the constant task-marginal term makes the value approximate ``-I``.

    Statistics update only in training mode, so validation and inference never
    change them. They are buffers and resume with checkpoints.

    Attributes:
        num_routers (int): Number of tasks ``R``.
        num_experts (int): Number of experts ``E``.
        warmup_updates (int): Training forwards that only accumulate statistics.
        momentum (float): Exponential moving-average momentum after warmup.
        joint (Tensor): Buffer with the ``p(task, expert)`` estimate. Shape: ``(R, E)``.
        updates (Tensor): Buffer counting statistics updates. Shape: ``()``.
    """

    joint: Tensor
    updates: Tensor

    def __init__(
        self, num_routers: int, num_experts: int, *, warmup_updates: int, momentum: float
    ) -> None:
        """Allocate zero statistics.

        Args:
            num_routers: Number of tasks ``R``.
            num_experts: Number of experts ``E``.
            warmup_updates: Forwards before the surrogate contributes a loss.
            momentum: Exponential moving-average momentum after warmup.
        """
        super().__init__()
        self.num_routers = num_routers
        self.num_experts = num_experts
        self.warmup_updates = warmup_updates
        self.momentum = momentum
        self.register_buffer("joint", torch.zeros(num_routers, num_experts))
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64))

    def forward(self, probabilities: Tensor, router_indices: Tensor) -> Tensor:
        """Update the stored joint estimate and return the differentiable surrogate.

        Args:
            probabilities (Tensor): FP32 routing probabilities over all experts.
                Shape: ``(B, T, E)``.
            router_indices (Tensor): Task of every sample. Shape: ``(B,)``.

        Returns:
            Tensor: Scalar FP32 loss, zero during warmup.
        """
        batch_size = probabilities.shape[0]
        sample_probabilities = probabilities.mean(dim=1)  # Shape: (B, E)
        batch_joint = (
            torch.zeros_like(self.joint).index_add(0, router_indices, sample_probabilities)
            / batch_size
        )  # Shape: (R, E); sums to one with the batch's task frequencies.
        with torch.no_grad():
            self.updates.add_(1)
            count = self.updates.to(self.joint.dtype)
            # A running mean during warmup gives every early batch equal weight;
            # tensor arithmetic avoids a host synchronization on the counter.
            in_warmup = self.updates <= self.warmup_updates
            momentum = torch.where(in_warmup, 1 - 1 / count, torch.full_like(count, self.momentum))
            self.joint.mul_(momentum).add_((1 - momentum) * batch_joint.detach())
            expert_marginal = self.joint.sum(dim=0)  # Shape: (E,)
            task_marginal = self.joint.sum(dim=1)  # Shape: (R,)
            log_ratio = (
                torch.log(self.joint + _LOG_EPSILON)
                - torch.log(expert_marginal + _LOG_EPSILON)[None, :]
                - torch.log(task_marginal + _LOG_EPSILON)[:, None]
            )  # Shape: (R, E), detached pointwise mutual information.
            active = (~in_warmup).to(batch_joint.dtype)
        return active * -(batch_joint * log_ratio).sum()

    def estimate(self) -> Tensor:
        """Return the stored mutual information ``I(T; E)`` in nats.

        Returns:
            Tensor: Scalar FP32 estimate from the joint buffer.
        """
        joint = self.joint
        total = joint.sum().clamp_min(_LOG_EPSILON)
        normalized = joint / total  # Shape: (R, E)
        independent = normalized.sum(dim=1, keepdim=True) * normalized.sum(dim=0, keepdim=True)
        terms = normalized * (
            torch.log(normalized + _LOG_EPSILON) - torch.log(independent + _LOG_EPSILON)
        )
        return terms.sum()


class MixtureOfExpertsFeedForward(nn.Module):
    """Replace a Transformer feed-forward block with task-routed sparse experts.

    Each expert is ``Linear(D, H) -> GELU -> Dropout -> Linear(H, D)``, matching the
    dense block it replaces. Each token is mixed from its top-k experts under its
    sample's task router. In training mode the layer stores its weighted
    mutual-information loss for the policy to collect after the flow loss. The loss
    is stored only for differentiable forwards, so one flow-matching loss evaluation
    must produce exactly one forward per layer.

    Attributes:
        dim (int): Token feature dimension ``D``.
        hidden_dim (int): Hidden width ``H`` of each expert.
        num_experts (int): Number of experts ``E``.
        top_k (int): Experts mixed per token ``K``.
        mutual_information_weight (float): Weight of this layer's surrogate loss.
        input_weight (nn.Parameter): Expert input weights, shape ``(E, D, H)``.
        input_bias (nn.Parameter): Expert input biases, shape ``(E, H)``.
        output_weight (nn.Parameter): Expert output weights, shape ``(E, H, D)``.
        output_bias (nn.Parameter): Expert output biases, shape ``(E, D)``.
        routers (TaskRouters): Task-specific expert scorers.
        mutual_information (TaskExpertMutualInformation): Routing statistics.
        dropout (nn.Dropout): Dropout after the expert activation.
        auxiliary_loss (Tensor | None): Weighted loss of the last training forward.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        num_experts: int,
        top_k: int,
        num_routers: int,
        dropout: float = 0.0,
        mutual_information_weight: float = 5e-4,
        mutual_information_warmup: int = 100,
        mutual_information_momentum: float = 0.99,
    ) -> None:
        """Allocate experts, routers and routing statistics.

        Args:
            dim: Token feature dimension ``D``.
            hidden_dim: Hidden width ``H`` of each expert.
            num_experts: Number of experts ``E``.
            top_k: Experts mixed per token; must not exceed ``num_experts``.
            num_routers: Number of task routers ``R``.
            dropout: Dropout probability after the expert activation.
            mutual_information_weight: Weight of the surrogate loss for this layer.
            mutual_information_warmup: Training forwards before the loss applies.
            mutual_information_momentum: Statistics momentum after warmup.
        """
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.mutual_information_weight = mutual_information_weight
        self.input_weight = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.input_bias = nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.output_weight = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.output_bias = nn.Parameter(torch.empty(num_experts, dim))
        self.reset_expert_parameters()
        self.routers = TaskRouters(dim, num_experts, num_routers)
        self.mutual_information = TaskExpertMutualInformation(
            num_routers,
            num_experts,
            warmup_updates=mutual_information_warmup,
            momentum=mutual_information_momentum,
        )
        self.dropout = nn.Dropout(dropout)
        self.auxiliary_loss: Tensor | None = None

    def reset_expert_parameters(self) -> None:
        """Initialize every expert independently like ``nn.Linear``'s default."""
        for weight, bias in (
            (self.input_weight, self.input_bias),
            (self.output_weight, self.output_bias),
        ):
            # nn.Linear's Kaiming-uniform(a=sqrt(5)) default reduces to U(+-1/sqrt(fan_in)).
            bound = 1 / math.sqrt(weight.shape[1])
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    def route(self, tokens: Tensor, router_indices: Tensor) -> tuple[Tensor, Tensor]:
        """Return FP32 routing probabilities and sparse top-k gates.

        Args:
            tokens (Tensor): Normalized token features. Shape: ``(B, T, D)``.
            router_indices (Tensor): Task router of every sample. Shape: ``(B,)``.

        Returns:
            tuple[Tensor, Tensor]: Probabilities over all experts and gates that keep
            only the top-k probabilities. Both shaped ``(B, T, E)``.
        """
        # Routing stays in FP32 so near-tied experts are ranked identically under AMP.
        probabilities = self.routers(tokens.float(), router_indices).float().softmax(dim=-1)
        top_probabilities, top_experts = probabilities.topk(self.top_k, dim=-1)
        # Shape: (B, T, K) each.
        gates = torch.zeros_like(probabilities).scatter(-1, top_experts, top_probabilities)
        return probabilities, gates  # Shape: (B, T, E) each.

    def forward(self, tokens: Tensor, router_indices: Tensor) -> Tensor:
        """Mix the top-k experts of every token under its sample's task router.

        Args:
            tokens (Tensor): Normalized token features. Shape: ``(B, T, D)``.
            router_indices (Tensor): Integer task router of every sample. Shape: ``(B,)``.

        Returns:
            Tensor: Mixed expert outputs. Shape: ``(B, T, D)``.
        """
        probabilities, gates = self.route(tokens, router_indices)
        hidden = torch.einsum("btd,edh->bteh", tokens, self.input_weight) + self.input_bias
        hidden = self.dropout(functional.gelu(hidden))  # Shape: (B, T, E, H)
        expert_outputs = (
            torch.einsum("bteh,ehd->bted", hidden, self.output_weight) + self.output_bias
        )  # Shape: (B, T, E, D)
        mixed = torch.einsum("bte,bted->btd", gates.to(expert_outputs.dtype), expert_outputs)
        # Shape: (B, T, D); unselected experts have exactly zero gate and gradient.
        # Only differentiable training forwards update statistics; no-grad sampling
        # from a model left in training mode must not count as a training batch.
        if self.training and torch.is_grad_enabled():
            if self.auxiliary_loss is not None:
                raise RuntimeError(
                    "Mixture-of-experts auxiliary loss was not collected before another "
                    "training forward"
                )
            self.auxiliary_loss = self.mutual_information_weight * self.mutual_information(
                probabilities, router_indices
            )
        return mixed

    def pop_auxiliary_loss(self) -> Tensor | None:
        """Return and clear the weighted loss of the last training forward.

        Returns:
            Tensor | None: Scalar FP32 loss, or None after an evaluation forward.
        """
        loss, self.auxiliary_loss = self.auxiliary_loss, None
        return loss
