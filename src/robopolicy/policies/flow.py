"""A single policy composes observation encoding, a head, and flow matching."""

from __future__ import annotations

from functools import partial
from typing import cast

import torch
from torch import Tensor, nn

from robopolicy.algorithms import build_algorithm
from robopolicy.algorithms.improved_mean_flow import ImprovedMeanFlowConfig
from robopolicy.contracts.batch import PolicyInput
from robopolicy.modules.observation import ObservationEncoder
from robopolicy.modules.policy_head.config import ConvHeadConfig
from robopolicy.modules.policy_head.transformer import (
    ExpertSettings,
    TransformerActionDecoder,
    TransformerActionDecoderConfig,
)
from robopolicy.modules.policy_head.unet import FiLMConvPolicyHead
from robopolicy.modules.queries import IndexedConditioning
from robopolicy.modules.transformer.moe import MixtureOfExpertsFeedForward
from robopolicy.policies.config import PolicyConfig


class ActionField(nn.Module):
    """Adapt two distinct head computations to the same flow-field signature."""

    def __init__(self, config: PolicyConfig, tokens: int) -> None:
        """Construct a head from settings owned by that component."""
        super().__init__()
        self.config = config
        head = config.head
        self.network: nn.Module
        if isinstance(head, ConvHeadConfig):
            self.network = FiLMConvPolicyHead(
                action_dim=config.action_dim,
                horizon=config.action_steps,
                global_cond_dim=tokens * config.fusion.dim,
                down_dims=head.down_dims,
                time_embed_dim=head.time_dim,
                kernel_size=head.kernel_size,
                n_groups=head.groups,
            )
        else:
            experts = None
            if head.moe is not None:
                conditioning = config.conditioning
                assert isinstance(conditioning, IndexedConditioning)
                experts = ExpertSettings(
                    experts=head.moe.experts,
                    top_k=head.moe.top_k,
                    hidden_dim=head.expert_ff_dim,
                    num_routers=conditioning.vocab_sizes[head.moe.router_slot],
                    mutual_information_weight=head.moe.mutual_information_weight,
                    mutual_information_warmup=head.moe.mutual_information_warmup,
                    mutual_information_momentum=head.moe.mutual_information_momentum,
                )
            self.network = TransformerActionDecoder(
                action_dim=config.action_dim,
                horizon=config.action_steps,
                context_dim=config.fusion.dim,
                robot_state_dim=config.fusion.dim,
                config=TransformerActionDecoderConfig(
                    d_model=head.dim,
                    n_layers=head.layers,
                    n_heads=head.heads,
                    ff_dim=head.ff_dim,
                    time_embed_dim=head.time_dim,
                    dropout=head.dropout,
                    attention_dropout=head.attention_dropout,
                    interval_conditioning=isinstance(config.algorithm, ImprovedMeanFlowConfig),
                    state_conditioning=head.state_conditioning,
                    experts=experts,
                ),
            )

    def forward(
        self,
        actions: Tensor,
        timestep: Tensor,
        global_cond: Tensor,
        *,
        time_delta: Tensor | None = None,
        router_indices: Tensor | None = None,
        context_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict velocities (B,T_action,D_action) from actions and times.

        Args:
            actions: Noisy actions ``(B, T_action, D_action)``.
            timestep: Flow times ``(B,)``.
            global_cond: Encoded observations ``(B, L_context, D)``.
            time_delta: Optional iMF intervals ``(B,)``.
            router_indices: Task routers ``(B,)`` required by sparse experts.
            context_key_padding_mask: Optional bool ``(B, L_context)`` over the encoder's
                tokens, True where an input mode hides them.

        Returns:
            Velocities ``(B, T_action, D_action)``.
        """
        if isinstance(self.config.head, ConvHeadConfig):
            if context_key_padding_mask is not None:
                raise ValueError("Input modes need the Transformer head's cross-attention")
            return cast(Tensor, self.network(actions, timestep, global_cond=global_cond))
        state = global_cond[:, : self.config.obs_steps]  # (B, T_obs, D_context)
        context = global_cond[:, self.config.obs_steps :]
        return cast(
            Tensor,
            self.network(
                actions,
                timestep,
                context=context,
                robot_state=state,
                time_delta=time_delta,
                router_indices=router_indices,
                context_key_padding_mask=None
                if context_key_padding_mask is None
                else context_key_padding_mask[:, self.config.obs_steps :],
            ),
        )

    def expert_layers(self) -> list[MixtureOfExpertsFeedForward]:
        """Return sparse-expert blocks in layer order; empty for dense heads."""
        if isinstance(self.network, TransformerActionDecoder):
            return self.network.expert_layers()
        return []


class FlowPolicy(nn.Module):
    """Public loss, prediction, and attention APIs sharing one observation encoder."""

    def __init__(self, config: PolicyConfig, initialize_pretrained: bool = False) -> None:
        """Construct the configured architecture, with no implicit downloads."""
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(
            config.state_dim,
            config.obs_steps,
            config.cameras,
            config.entity_dim,
            config.selected_entities,
            config.conditioning,
            config.fusion,
            config.vision,
            initialize_pretrained,
        )
        self.field = ActionField(config, self.encoder.token_count)
        self.algorithm = build_algorithm(config.algorithm)

    def encode(
        self, inputs: PolicyInput, return_attention: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Validate and encode canonical observations into ``(B, L_context, D)``."""
        return cast(
            tuple[Tensor, dict[str, Tensor]],
            self.encoder(inputs, return_attention=return_attention),
        )

    def router_indices(self, inputs: PolicyInput) -> Tensor | None:
        """Select each sample's expert router from its configured task slot.

        Args:
            inputs: Canonical observations; indices were validated by the encoder.

        Returns:
            Integer routers ``(B,)``, or None for heads without sparse experts.
        """
        head = self.config.head
        if isinstance(head, ConvHeadConfig) or head.moe is None:
            return None
        assert inputs.task_indices is not None
        return inputs.task_indices[:, head.moe.router_slot]  # Shape: (B,)

    def field_kwargs(self, inputs: PolicyInput) -> dict[str, Tensor]:
        """Per-sample field arguments: the expert router ``(B,)`` and the context tokens an
        input mode hides ``(B, L_context)``, each only when the policy uses them. Callers that
        repeat the context per sample repeat these alongside it."""
        kwargs: dict[str, Tensor] = {}
        routers = self.router_indices(inputs)
        if routers is not None:
            kwargs["router_indices"] = routers
        mask = self.encoder.context_padding_mask(inputs)
        if mask is not None:
            kwargs["context_key_padding_mask"] = mask
        return kwargs

    def routed_field(self, inputs: PolicyInput) -> partial[Tensor]:
        """Bind each sample's router and hidden tokens so algorithms see the shared field signature."""
        return partial(self.field, **self.field_kwargs(inputs))

    def loss(self, inputs: PolicyInput) -> Tensor:
        """Return the scalar training objective using actions ``(B, T_action, D_action)``."""
        return self.loss_terms(inputs)["loss"]

    def loss_terms(self, inputs: PolicyInput) -> dict[str, Tensor]:
        """Return the total objective and its scalar components.

        Args:
            inputs: Canonical observations with target actions.

        Returns:
            ``loss``, the optimized total. Sparse-expert heads add its components
            ``flow_loss`` and ``moe_mutual_information_loss``; the latter is zero
            outside differentiable training forwards.
        """
        actions = inputs.actions
        if actions is None or actions.shape != (
            inputs.state.shape[0],
            self.config.action_steps,
            self.config.action_dim,
        ):
            raise ValueError("Training requires actions matching (B, T_action, D_action)")
        if not actions.is_floating_point():
            raise ValueError("Actions must be floating point")
        experts = self.field.expert_layers()
        for layer in experts:
            layer.pop_auxiliary_loss()  # Discard losses left by an interrupted step.
        context, _ = self.encode(inputs)
        flow_loss = self.algorithm.compute_loss(self.routed_field(inputs), actions, context)
        if not experts:
            return {"loss": flow_loss}
        layer_losses = [layer.pop_auxiliary_loss() for layer in experts]
        expert_loss = sum(
            (loss for loss in layer_losses if loss is not None),
            torch.zeros((), device=flow_loss.device),
        )
        return {
            "loss": flow_loss + expert_loss,
            "flow_loss": flow_loss,
            "moe_mutual_information_loss": expert_loss,
        }

    def routing_metrics(self) -> dict[str, float]:
        """Summarize stored task-expert statistics without a forward pass.

        Returns:
            Per layer: the mutual information ``I(task; expert)`` in nats, and per
            task the effective number of experts ``exp(H(expert | task))``.
        """
        metrics: dict[str, float] = {}
        for index, layer in enumerate(self.field.expert_layers()):
            statistics = layer.mutual_information
            joint = statistics.joint.detach().float()  # Shape: (R, E)
            metrics[f"moe/layer{index}/mutual_information"] = float(statistics.estimate())
            conditional = joint / joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
            entropy = -(conditional * torch.log(conditional.clamp_min(1e-12))).sum(dim=1)
            for task, value in enumerate(torch.exp(entropy).tolist()):
                metrics[f"moe/layer{index}/task{task}_effective_experts"] = value
        return metrics

    @torch.no_grad()
    def predict(self, inputs: PolicyInput, noise: Tensor | None = None) -> Tensor:
        """Generate normalized actions ``(B, T_action, D_action)`` from observations.

        Optional fixed noise makes prediction parity directly testable. Callers
        choose model mode; inference checkpoint loading returns an eval model.
        """
        self.encoder.validate(inputs)
        shape = (inputs.state.shape[0], self.config.action_steps, self.config.action_dim)
        if noise is None:
            noise = torch.randn(shape, device=inputs.state.device, dtype=inputs.state.dtype)
        if (
            noise.shape != shape
            or noise.device != inputs.state.device
            or noise.dtype != inputs.state.dtype
        ):
            raise ValueError("Initial noise must match action shape, input device, and dtype")
        return self.predict_tensors(inputs, noise)

    def predict_tensors(self, inputs: PolicyInput, noise: Tensor) -> Tensor:
        """Run the shared prediction computation after boundary validation.

        Args:
            inputs: Validated canonical observations, already on the model device.
            noise: Initial ``(B, T_action, D_action)`` actions in the solver dtype.

        Returns:
            Normalized actions ``(B, T_action, D_action)``. Keeping the solver
            state in the noise dtype preserves FP32 accumulation under autocast.
        """
        context, _ = self.encoder.encode_tensors(inputs)
        return self.algorithm.generate_samples(self.routed_field(inputs), noise, context)

    def attention_maps(self, inputs: PolicyInput) -> dict[str, Tensor]:
        """Return maps from the same encoding path used by loss and prediction."""
        return self.encode(inputs, return_attention=True)[1]


def build_policy(config: PolicyConfig, *, initialize_pretrained: bool = False) -> FlowPolicy:
    """Build any supported policy for training or inference from the same factory."""
    return FlowPolicy(config, initialize_pretrained)
