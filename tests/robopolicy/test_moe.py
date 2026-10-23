"""Task-routed sparse experts, their mutual-information objective, and policy wiring."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional

from robopolicy.contracts.batch import PolicyInput
from robopolicy.modules.transformer.moe import (
    MixtureOfExpertsFeedForward,
    TaskExpertMutualInformation,
)
from robopolicy.policies.config import PolicyConfig
from robopolicy.policies.flow import build_policy


def expert_layer(**kwargs):
    """Create a small layer with distinct routers and experts."""
    options = {
        "num_experts": 4,
        "top_k": 2,
        "num_routers": 3,
        "mutual_information_warmup": 0,
        **kwargs,
    }
    return MixtureOfExpertsFeedForward(8, 6, **options)


def reference_output(layer, tokens, routers):
    """Evaluate the sparse mixture token by token with explicit expert calls."""
    outputs = torch.zeros_like(tokens)
    for sample in range(tokens.shape[0]):
        router = int(routers[sample])
        for step in range(tokens.shape[1]):
            token = tokens[sample, step]
            hidden = functional.gelu(
                token @ layer.routers.hidden_weight[router] + layer.routers.hidden_bias[router]
            )
            logits = (
                hidden @ layer.routers.output_weight[router] + layer.routers.output_bias[router]
            )
            probabilities = logits.softmax(dim=-1)
            values, experts = probabilities.topk(layer.top_k)
            for gate, expert in zip(values, experts):
                expert_hidden = functional.gelu(
                    token @ layer.input_weight[expert] + layer.input_bias[expert]
                )
                outputs[sample, step] += gate * (
                    expert_hidden @ layer.output_weight[expert] + layer.output_bias[expert]
                )
    return outputs


def test_dense_dispatch_matches_per_token_sparse_reference():
    torch.manual_seed(0)
    layer = expert_layer().eval()
    with torch.no_grad():
        # Larger router weights make the selection nontrivial for every token.
        layer.routers.output_weight.mul_(50)
    tokens = torch.randn(5, 3, 8)
    routers = torch.tensor([0, 2, 1, 2, 0])
    torch.testing.assert_close(layer(tokens, routers), reference_output(layer, tokens, routers))
    _, gates = layer.route(tokens, routers)
    assert ((gates > 0).sum(dim=-1) == 2).all()


def test_only_selected_experts_and_used_routers_receive_gradients():
    torch.manual_seed(1)
    layer = expert_layer(mutual_information_weight=0.0)
    with torch.no_grad():
        layer.routers.output_weight.mul_(50)
    tokens = torch.randn(4, 2, 8)
    routers = torch.tensor([1, 1, 1, 1])
    layer(tokens, routers).square().sum().backward()
    _, gates = layer.route(tokens, routers)
    selected = (gates > 0).flatten(0, 1).any(dim=0)  # Shape: (E,)
    for expert in range(layer.num_experts):
        has_gradient = bool(layer.input_weight.grad[expert].abs().sum() > 0)
        assert has_gradient == bool(selected[expert])
    assert layer.routers.hidden_weight.grad[1].abs().sum() > 0
    assert layer.routers.hidden_weight.grad[[0, 2]].abs().sum() == 0
    assert layer.routers.output_weight.grad[[0, 2]].abs().sum() == 0


def test_mixed_task_samples_are_routed_independently():
    torch.manual_seed(2)
    layer = expert_layer().eval()
    tokens = torch.randn(4, 3, 8)
    routers = torch.tensor([0, 1, 0, 1])
    original = layer(tokens, routers)
    with torch.no_grad():
        layer.routers.output_weight[1].normal_(std=5)
    changed = layer(tokens, routers)
    torch.testing.assert_close(changed[routers == 0], original[routers == 0])
    assert not torch.allclose(changed[routers == 1], original[routers == 1])


def test_statistics_warm_up_as_running_mean_then_use_momentum():
    statistics = TaskExpertMutualInformation(2, 2, warmup_updates=2, momentum=0.75)
    first = torch.tensor([[[0.9, 0.1]], [[0.2, 0.8]]])  # (B=2, T=1, E=2)
    second = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]])
    routers = torch.tensor([0, 1])
    assert statistics(first, routers) == 0
    assert statistics(second, routers) == 0
    joint_first = torch.tensor([[0.45, 0.05], [0.10, 0.40]])
    joint_second = torch.full((2, 2), 0.25)
    torch.testing.assert_close(statistics.joint, (joint_first + joint_second) / 2)
    loss = statistics(first, routers)
    expected_joint = 0.75 * (joint_first + joint_second) / 2 + 0.25 * joint_first
    torch.testing.assert_close(statistics.joint, expected_joint)
    pointwise = torch.log(expected_joint + 1e-4) - (
        torch.log(expected_joint.sum(0) + 1e-4)[None]
        + torch.log(expected_joint.sum(1) + 1e-4)[:, None]
    )
    torch.testing.assert_close(loss, -(joint_first * pointwise).sum())
    assert statistics.updates == 3


def test_surrogate_ascent_separates_tasks_onto_experts():
    torch.manual_seed(3)
    statistics = TaskExpertMutualInformation(2, 2, warmup_updates=1, momentum=0.9)
    logits = torch.zeros(2, 2, requires_grad=True)  # One routing distribution per task.
    optimizer = torch.optim.SGD([logits], lr=5.0)
    routers = torch.tensor([0, 1])
    for _ in range(200):
        optimizer.zero_grad()
        probabilities = (logits + 0.01 * torch.randn(2, 2)).softmax(-1)[:, None]  # (B, 1, E)
        statistics(probabilities, routers).backward()
        optimizer.step()
    # Two equally frequent tasks on disjoint experts carry log(2) nats.
    assert statistics.estimate() > 0.9 * torch.log(torch.tensor(2.0))


def test_statistics_change_only_in_differentiable_training_forwards():
    layer = expert_layer()
    tokens, routers = torch.randn(2, 3, 8), torch.tensor([0, 2])
    layer.eval()
    layer(tokens, routers)
    assert layer.mutual_information.updates == 0 and layer.pop_auxiliary_loss() is None
    layer.train()
    with torch.no_grad():
        layer(tokens, routers)
    assert layer.mutual_information.updates == 0 and layer.pop_auxiliary_loss() is None
    layer(tokens, routers)
    assert layer.mutual_information.updates == 1
    with pytest.raises(RuntimeError, match="not collected"):
        layer(tokens, routers)
    assert layer.pop_auxiliary_loss() is not None
    assert {"mutual_information.joint", "mutual_information.updates"} <= set(layer.state_dict())


def moe_config(**moe):
    """Create a tiny indexed policy whose head uses sparse experts."""
    return PolicyConfig(
        state_dim=2,
        action_dim=1,
        obs_steps=1,
        action_steps=4,
        conditioning={"type": "indexed", "vocab_sizes": [2, 3]},
        fusion={"dim": 16, "heads": 2, "layers": 1, "ff_dim": 32},
        head={
            "type": "transformer",
            "dim": 16,
            "heads": 2,
            "layers": 2,
            "ff_dim": 32,
            "time_dim": 8,
            "moe": {"experts": 4, "top_k": 2, "router_slot": 1, **moe},
        },
    )


def test_policy_routes_by_slot_and_reports_loss_components():
    torch.manual_seed(4)
    policy = build_policy(moe_config(mutual_information_warmup=1))
    layers = policy.field.expert_layers()
    assert len(layers) == 2 and layers[0].routers.num_routers == 3
    assert layers[0].hidden_dim == 16  # ff_dim // top_k keeps the active width dense-sized.
    inputs = PolicyInput(
        state=torch.randn(6, 1, 2),
        task_indices=torch.tensor([[0, 0], [1, 1], [0, 2], [1, 0], [0, 1], [1, 2]]),
        actions=torch.randn(6, 4, 1),
    )
    torch.testing.assert_close(policy.router_indices(inputs), inputs.task_indices[:, 1])
    warmup = policy.loss_terms(inputs)
    assert set(warmup) == {"loss", "flow_loss", "moe_mutual_information_loss"}
    assert warmup["moe_mutual_information_loss"] == 0
    terms = policy.loss_terms(inputs)
    assert terms["moe_mutual_information_loss"] != 0
    torch.testing.assert_close(
        terms["loss"], terms["flow_loss"] + terms["moe_mutual_information_loss"]
    )
    terms["loss"].backward()
    assert all(layer.routers.hidden_weight.grad.abs().sum() > 0 for layer in layers)
    metrics = policy.routing_metrics()
    assert set(metrics) == {
        f"moe/layer{layer}/{name}"
        for layer in range(2)
        for name in ("mutual_information", *(f"task{task}_effective_experts" for task in range(3)))
    }
    policy.eval()
    assert policy.loss_terms(inputs)["moe_mutual_information_loss"] == 0
    observation = replace(inputs, actions=None)
    prediction = policy.predict(observation, torch.zeros(6, 4, 1))
    assert prediction.shape == (6, 4, 1) and torch.isfinite(prediction).all()


def test_dense_heads_report_only_the_total_loss():
    config = moe_config()
    dense = PolicyConfig.model_validate(
        {**config.model_dump(), "head": {**config.head.model_dump(), "moe": None}}
    )
    policy = build_policy(dense)
    assert policy.field.expert_layers() == [] and policy.routing_metrics() == {}
    inputs = PolicyInput(
        state=torch.randn(2, 1, 2),
        task_indices=torch.zeros((2, 2), dtype=torch.int64),
        actions=torch.randn(2, 4, 1),
    )
    assert set(policy.loss_terms(inputs)) == {"loss"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"conditioning": {"type": "learned", "num_tokens": 1}}, "indexed conditioning"),
        ({"algorithm": {"type": "improved_mean_flow"}}, "flow matching only"),
        ({"head_moe": {"router_slot": 2}}, "router_slot"),
        ({"head_moe": {"top_k": 5}}, "top_k"),
    ],
)
def test_invalid_expert_configurations_are_rejected(change, message):
    values = moe_config().model_dump()
    if "head_moe" in change:
        values["head"]["moe"].update(change.pop("head_moe"))
    values.update(change)
    with pytest.raises(ValueError, match=message):
        PolicyConfig.model_validate(values)
