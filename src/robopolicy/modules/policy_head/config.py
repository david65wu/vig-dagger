"""Action-head architecture settings, composed by PolicyConfig."""

from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from robopolicy.contracts.schema import StrictConfig


class ConvHeadConfig(StrictConfig):
    """FiLM-conditioned convolutional U-Net settings."""

    type: Literal["conv"] = "conv"
    down_dims: tuple[int, ...] = (64, 128, 256)
    time_dim: int = Field(default=64, ge=4, multiple_of=2)
    kernel_size: int = Field(default=5, ge=1)
    groups: int = Field(default=8, ge=1)


class MixtureOfExpertsConfig(StrictConfig):
    """Sparse Diffusion Policy experts replacing every Transformer feed-forward block.

    Attributes:
        experts: Experts per layer.
        top_k: Experts mixed per token.
        expert_ff_dim: Hidden width of each expert; defaults to ``ff_dim // top_k``
            so the active feed-forward width equals the dense head's.
        router_slot: ``task_indices`` slot whose vocabulary selects one router per
            category; its vocabulary size is the number of routers.
        mutual_information_weight: Task-expert mutual-information weight per layer.
        mutual_information_warmup: Training forwards that only gather statistics.
        mutual_information_momentum: Statistics moving-average momentum.
    """

    experts: int = Field(default=8, ge=2)
    top_k: int = Field(default=2, ge=1)
    expert_ff_dim: int | None = Field(default=None, ge=1)
    router_slot: int = Field(default=0, ge=0)
    mutual_information_weight: float = Field(default=5e-4, ge=0)
    mutual_information_warmup: int = Field(default=100, ge=0)
    mutual_information_momentum: float = Field(default=0.99, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_top_k(self) -> Self:
        """Reject selecting more experts than a layer contains."""
        if self.top_k > self.experts:
            raise ValueError("Mixture-of-experts top_k cannot exceed experts")
        return self


class TransformerHeadConfig(StrictConfig):
    """Adaptive LayerNorm Transformer action decoder settings."""

    type: Literal["transformer"] = "transformer"
    dim: int = Field(default=128, ge=8)
    layers: int = Field(default=4, ge=1)
    heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=256, ge=1)
    time_dim: int = Field(default=64, ge=4, multiple_of=2)
    dropout: float = Field(default=0, ge=0, lt=1)
    attention_dropout: float = Field(default=0, ge=0, lt=1)
    state_conditioning: Literal["mean", "current"] = "mean"
    moe: MixtureOfExpertsConfig | None = None

    @model_validator(mode="after")
    def validate_expert_width(self) -> Self:
        """Require a positive default expert width."""
        if self.moe is not None and self.moe.expert_ff_dim is None:
            if self.ff_dim < self.moe.top_k:
                raise ValueError("ff_dim must be at least top_k for the default expert width")
        return self

    @property
    def expert_ff_dim(self) -> int:
        """Return each expert's hidden width; requires configured experts."""
        if self.moe is None:
            raise ValueError("This head has no mixture of experts")
        return self.moe.expert_ff_dim or self.ff_dim // self.moe.top_k


HeadConfig = Annotated[ConvHeadConfig | TransformerHeadConfig, Field(discriminator="type")]
