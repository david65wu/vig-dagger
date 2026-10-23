"""Input modes: per-sample hidden camera groups as a categorical input shared by training and inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from robopolicy import RunConfig, Trainer, load_policy
from robopolicy.contracts.schema import Feature, FeatureSchema
from robopolicy.processing.input_modes import InputModesConfig
from robopolicy.processing.normalization import Statistics
from robopolicy.processing.packing import InputPacker, ProcessingConfig
from tests.robopolicy.test_dataset import write_dataset

MODES = {"hidden": [[], ["right"], ["left"]]}  # all cameras, the right one hidden, the left one hidden


def packer() -> InputPacker:
    """Bind an input-mode packer with two cameras and one dataset task slot to unit statistics."""
    schema = FeatureSchema(
        features=(
            Feature(key="state", shape=(1,), paths=("state",)),
            Feature(key="actions", shape=(1,), paths=("actions",)),
            Feature(key="images", shape=(3, 16, 16), source="video", cameras=("right", "left"), dtype="uint8"),
            Feature(key="task_indices", shape=(1,), paths=("i",), dtype="int64"),
        )
    )
    statistics = Statistics(feature_schema=schema, dataset_fingerprint="test", selection=(), features={})
    return InputPacker(schema, statistics, ProcessingConfig(input_modes=MODES))


def raw_batch(batch: int) -> dict[str, torch.Tensor]:
    return {
        "state": torch.zeros(batch, 1, 1),
        "images": torch.randint(0, 256, (batch, 1, 2, 3, 16, 16), dtype=torch.uint8),
        "task_indices": torch.full((batch, 1, 1), 5, dtype=torch.int64),
    }


def test_config_rejects_bad_modes():
    with pytest.raises(ValueError, match="more than once"):
        InputModesConfig(hidden=[[], ["a", "a"]])
    with pytest.raises(ValueError, match="different camera sets"):
        InputModesConfig(hidden=[["a"], ["a"]])
    with pytest.raises(ValueError, match="either resolution levels or input modes"):
        ProcessingConfig(resolution={"factors": [1, 2]}, input_modes=MODES)
    schema = FeatureSchema(
        features=(
            Feature(key="state", shape=(1,), paths=("state",)),
            Feature(key="images", shape=(3, 16, 16), source="video", cameras=("right",), dtype="uint8"),
        )
    )
    with pytest.raises(ValueError, match="unknown cameras"):
        ProcessingConfig(input_modes=MODES).validate_schema(schema)


def test_packer_draws_cycles_or_requires_modes():
    mode_packer = packer()
    batch = raw_batch(6)
    torch.manual_seed(0)
    training = mode_packer(batch, training=True)
    assert training.task_indices.shape == (6, 2) and (training.task_indices[:, 0] == 5).all()
    modes = training.task_indices[:, 1]
    assert ((modes >= 0) & (modes < 3)).all() and len(modes.unique()) > 1
    table = torch.tensor([[False, False], [True, False], [False, True]])
    assert torch.equal(training.camera_mask, table[modes])
    validation = mode_packer({**batch, "actions": torch.zeros(6, 1, 1)}, require_actions=True)
    assert validation.task_indices[:, 1].tolist() == [0, 1, 2, 0, 1, 2]
    with pytest.raises(ValueError, match="require int64 mode_indices"):
        mode_packer.validate(batch)
    inference = mode_packer({**batch, "mode_indices": torch.full((6, 1), 2)})
    assert (inference.task_indices[:, 1] == 2).all() and inference.camera_mask[:, 1].all()
    assert not inference.camera_mask[:, 0].any()
    for invalid in (torch.tensor([[3]] * 6), torch.tensor([2] * 6), torch.tensor([[1.0]] * 6)):
        with pytest.raises(ValueError, match="mode_indices"):
            mode_packer.validate({**batch, "mode_indices": invalid})


def modes_run(tmp_path: Path, vocab_sizes=(1, 1, 3)) -> RunConfig:
    """Configure a tiny two-camera video run with experts routed by input mode."""
    schema = write_dataset(tmp_path / "data", lengths=((8, 9, 10, 11, 12),), video=True)
    return RunConfig(
        features=schema,
        dataset={"root": str(tmp_path / "data"), "video": {"decoder": "opencv"}, "selection": {"val_fraction": 0.3}},
        loader={"batch_size": 4},
        policy={
            "state_dim": 1,
            "action_dim": 1,
            "obs_steps": 3,
            "action_steps": 3,
            "cameras": 2,
            "vision": {"type": "patch", "patch_size": 4, "dim": 16},
            "conditioning": {"type": "indexed", "vocab_sizes": list(vocab_sizes)},
            "fusion": {"dim": 16, "heads": 2, "layers": 1, "ff_dim": 32},
            "head": {
                "type": "transformer",
                "dim": 16,
                "heads": 2,
                "layers": 2,
                "ff_dim": 32,
                "time_dim": 8,
                "moe": {"experts": 4, "top_k": 2, "router_slot": 2, "mutual_information_warmup": 2},
            },
        },
        processing={"input_modes": MODES},
        training={"device": "cpu", "max_updates": 6, "checkpoint_every": 3, "validate_every": 3, "log_every": 1},
        logging={"out_dir": str(tmp_path / "run"), "wandb": {"enabled": False}},
    )


def test_run_config_reserves_the_last_slot_for_modes(tmp_path):
    config = modes_run(tmp_path)
    with pytest.raises(ValueError, match="Input modes require indexed conditioning"):
        RunConfig.model_validate(
            {
                **config.model_dump(),
                "policy": {**config.policy.model_dump(), "conditioning": {"type": "indexed", "vocab_sizes": [1, 1, 4]}},
            }
        )


def test_hidden_cameras_cannot_change_the_actions(tmp_path):
    Trainer(modes_run(tmp_path)).fit()
    events = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines() if '"event": "train"' in line]
    assert len(events) == 6 and "moe/layer1/task2_effective_experts" in events[-1]
    inference = load_policy(tmp_path / "run/latest.pt", device="cpu")
    torch.manual_seed(1)
    observation = {
        "state": torch.zeros(2, 3, 1),
        "images": torch.randint(0, 256, (2, 3, 2, 3, 16, 16), dtype=torch.uint8),
        "task_indices": torch.zeros(2, 1, 2, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="mode_indices"):
        inference.predict(observation)
    other_right = {**observation, "images": observation["images"].clone()}
    other_right["images"][:, :, 0] = torch.randint(0, 256, (2, 3, 3, 16, 16), dtype=torch.uint8)
    noise = torch.zeros(2, 3, 1)

    def act(obs, mode):
        return inference.predict({**obs, "mode_indices": torch.full((2, 1), mode)}, noise)

    # Mode 1 hides the right camera: its images cannot move the actions; with every camera visible they do.
    torch.testing.assert_close(act(observation, 1), act(other_right, 1), rtol=0, atol=1e-6)
    assert not torch.allclose(act(observation, 0), act(other_right, 0), rtol=0, atol=1e-6)
    assert not torch.allclose(act(observation, 2), act(other_right, 2), rtol=0, atol=1e-6)
    assert all(torch.isfinite(act(observation, m)).all() for m in range(3))
