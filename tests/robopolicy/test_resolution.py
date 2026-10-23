"""Resolution degradation as a categorical input shared by training and inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robopolicy import RunConfig, Trainer, load_policy
from robopolicy.checkpoints.serialization import read_checkpoint
from robopolicy.contracts.schema import Feature, FeatureSchema
from robopolicy.processing.normalization import Statistics
from robopolicy.processing.packing import InputPacker, ProcessingConfig
from robopolicy.processing.resolution import ResolutionConfig, degrade_resolution
from tests.robopolicy.test_dataset import write_dataset

FACTORS = (1, 2, 4, 8)


def test_degradation_matches_opencv_round_trip():
    import cv2

    generator = np.random.default_rng(0)
    # Smooth structure plus texture resembles rendered frames better than pure noise.
    base = generator.integers(0, 256, (3, 8, 8, 3)).astype(np.uint8)
    frames = np.stack(
        [cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC) for image in base]
    )
    frames = np.clip(frames + generator.integers(-20, 21, frames.shape), 0, 255).astype(np.uint8)
    images = torch.from_numpy(frames).permute(0, 3, 1, 2)[:, None, None]  # (3, 1, 1, 3, 64, 64)
    for index, factor in enumerate(FACTORS):
        degraded = degrade_resolution(images, torch.full((3,), index), FACTORS)
        assert degraded.dtype == torch.uint8 and degraded.shape == images.shape
        if factor == 1:
            assert torch.equal(degraded, images)
            continue
        expected = np.stack(
            [
                cv2.resize(
                    cv2.resize(frame, (64 // factor, 64 // factor), interpolation=cv2.INTER_AREA),
                    (64, 64),
                    interpolation=cv2.INTER_LINEAR,
                )
                for frame in frames
            ]
        )
        actual = degraded[:, 0, 0].permute(0, 2, 3, 1).numpy()
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        # OpenCV's fixed-point arithmetic rounds a few pixels differently.
        assert difference.max() <= 2 and difference.mean() < 0.2


def test_each_sample_uses_its_own_factor_for_all_frames():
    images = torch.randint(0, 256, (3, 2, 2, 3, 16, 16), dtype=torch.uint8)
    degraded = degrade_resolution(images, torch.tensor([3, 0, 1]), FACTORS)
    assert torch.equal(degraded[1], images[1])
    for sample, index in ((0, 3), (2, 1)):
        alone = degrade_resolution(images[sample : sample + 1], torch.tensor([index]), FACTORS)
        assert torch.equal(degraded[sample : sample + 1], alone)
    # A factor-8 frame is constant within each restored 8x8 block interior.
    assert degraded[0].float().std() < images[0].float().std()


def packer(task_slots: bool = False) -> InputPacker:
    """Bind a resolution packer to unit statistics."""
    features = [
        Feature(key="state", shape=(1,), paths=("state",)),
        Feature(key="actions", shape=(1,), paths=("actions",)),
        Feature(key="images", shape=(3, 16, 16), source="video", cameras=("a",), dtype="uint8"),
    ]
    if task_slots:
        features.append(Feature(key="task_indices", shape=(1,), paths=("i",), dtype="int64"))
    schema = FeatureSchema(features=tuple(features))
    statistics = Statistics(
        feature_schema=schema, dataset_fingerprint="test", selection=(), features={}
    )
    return InputPacker(schema, statistics, ProcessingConfig(resolution=ResolutionConfig()))


def raw_batch(batch: int, task_slots: bool = False) -> dict[str, torch.Tensor]:
    """Return uint8 images and float state for a resolution packer."""
    values = {
        "state": torch.zeros(batch, 1, 1),
        "images": torch.randint(0, 256, (batch, 1, 1, 3, 16, 16), dtype=torch.uint8),
    }
    if task_slots:
        values["task_indices"] = torch.full((batch, 1, 1), 5, dtype=torch.int64)
    return values


def test_packer_draws_cycles_or_requires_levels():
    resolution_packer = packer(task_slots=True)
    batch = raw_batch(6, task_slots=True)
    torch.manual_seed(0)
    training = resolution_packer(batch, training=True, require_actions=False)
    assert training.task_indices.shape == (6, 2)
    assert (training.task_indices[:, 0] == 5).all()
    levels = training.task_indices[:, 1]
    assert ((levels >= 0) & (levels < 4)).all() and len(levels.unique()) > 1
    validation = resolution_packer({**batch, "actions": torch.zeros(6, 1, 1)}, require_actions=True)
    assert validation.task_indices[:, 1].tolist() == [0, 1, 2, 3, 0, 1]
    with pytest.raises(ValueError, match="require int64 resolution_indices"):
        resolution_packer.validate(batch)
    supplied = {**batch, "resolution_indices": torch.tensor([[3]] * 6)}
    inference = resolution_packer(supplied)
    assert (inference.task_indices[:, 1] == 3).all()
    expected = degrade_resolution(batch["images"], torch.full((6,), 3), FACTORS)
    reference = resolution_packer(
        {**batch, "images": expected, "resolution_indices": torch.zeros(6, 1, dtype=torch.int64)}
    )
    # Degrading inside packing equals packing an already degraded image at factor 1.
    torch.testing.assert_close(inference.images, reference.images)
    for invalid in (torch.tensor([[4]] * 6), torch.tensor([3] * 6), torch.tensor([[1.0]] * 6)):
        with pytest.raises(ValueError, match="resolution_indices"):
            resolution_packer.validate({**batch, "resolution_indices": invalid})
    assert torch.equal(supplied["images"], batch["images"])  # Caller storage is unchanged.


def test_resolution_requires_divisible_images():
    schema = FeatureSchema(
        features=(
            Feature(key="state", shape=(1,), paths=("state",)),
            Feature(key="images", shape=(3, 12, 12), source="video", cameras=("a",), dtype="uint8"),
        )
    )
    with pytest.raises(ValueError, match="divisible"):
        ProcessingConfig(resolution=ResolutionConfig()).validate_schema(schema)
    with pytest.raises(ValueError, match="unique positive"):
        ResolutionConfig(factors=(1, 2, 2))


def resolution_run(tmp_path: Path, vocab_sizes=(1, 1, 4)) -> RunConfig:
    """Configure a tiny video run with experts routed by resolution level."""
    schema = write_dataset(tmp_path / "data", lengths=((8, 9, 10, 11, 12),), video=True)
    return RunConfig(
        features=schema,
        dataset={
            "root": str(tmp_path / "data"),
            "video": {"decoder": "opencv"},
            "selection": {"val_fraction": 0.3},
        },
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
        processing={"resolution": {"factors": list(FACTORS)}},
        training={
            "device": "cpu",
            "max_updates": 6,
            "checkpoint_every": 3,
            "validate_every": 3,
            "log_every": 1,
        },
        logging={"out_dir": str(tmp_path / "run"), "wandb": {"enabled": False}},
    )


def test_run_config_reserves_the_last_slot_for_resolution(tmp_path):
    config = resolution_run(tmp_path)
    with pytest.raises(ValueError, match="last slot"):
        RunConfig.model_validate(
            {
                **config.model_dump(),
                "policy": {
                    **config.policy.model_dump(),
                    "conditioning": {"type": "indexed", "vocab_sizes": [1, 1, 3]},
                },
            }
        )
    with pytest.raises(ValueError, match="dataset vocabulary slots"):
        RunConfig.model_validate(
            {
                **config.model_dump(),
                "policy": {
                    **config.policy.model_dump(),
                    "conditioning": {"type": "indexed", "vocab_sizes": [1, 4]},
                    "head": {
                        **config.policy.head.model_dump(),
                        "moe": {**config.policy.head.moe.model_dump(), "router_slot": 1},
                    },
                },
            }
        )


def test_training_logs_experts_and_inference_requires_levels(tmp_path):
    Trainer(resolution_run(tmp_path)).fit()
    events = [
        json.loads(line)
        for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
        if '"event": "train"' in line
    ]
    assert len(events) == 6
    assert all(
        {"train/flow_loss", "train/moe_mutual_information_loss"} <= set(event) for event in events
    )
    assert events[0]["train/moe_mutual_information_loss"] == 0  # Warmup.
    assert events[-1]["train/moe_mutual_information_loss"] != 0
    assert "moe/layer1/task3_effective_experts" in events[-1]
    payload = read_checkpoint(tmp_path / "run/latest.pt")
    assert payload["model"]["field.network.blocks.0.ff.mutual_information.updates"] == 6

    inference = load_policy(tmp_path / "run/latest.pt", device="cpu")
    observation = {
        "state": torch.zeros(2, 3, 1),
        "images": torch.randint(0, 256, (2, 3, 2, 3, 16, 16), dtype=torch.uint8),
        "task_indices": torch.zeros(2, 1, 2, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="resolution_indices"):
        inference.predict(observation)
    noise = torch.zeros(2, 3, 1)
    actions = {
        level: inference.predict(
            {**observation, "resolution_indices": torch.full((2, 1), level)}, noise
        )
        for level in range(4)
    }
    assert all(
        value.shape == (2, 3, 1) and torch.isfinite(value).all() for value in actions.values()
    )
    assert not torch.equal(actions[0], actions[3])
