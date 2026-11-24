"""Iteration mode (configs/plugcharger.yaml on this branch): rounds of DAgger, one MoE per round, adaptive threshold."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vig_dagger import adaptive, plan, steps, tasks, train
from vig_dagger import config as config_module
from vig_dagger.config import Collection, Paths, collections, target_of
from vig_dagger.dataset import training_pieces

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def it_cfg(tmp_path):
    return config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={tmp_path / 'work'}"])


def test_config_and_collections(it_cfg):
    cols = collections(it_cfg)
    assert [c.id for c in cols] == [f"round{r}_sd0_vig_adaptive_random_move" for r in (1, 2, 3, 4)]
    assert [target_of(it_cfg, c) for c in cols] == [100, 200, 400, 800]
    assert [c.moe for c in cols] == ["round1", "round2", "round3", "round4"]
    assert all(Collection.parse(c.id) == c for c in cols)
    assert it_cfg["dagger"]["expert"] == "fast_maniskill" and it_cfg["demos"]["motion"] == "fast_maniskill"
    assert it_cfg["dagger"]["save_failed_trajectories"] is True
    assert Paths.of(it_cfg).moe_dir("round2").name == "round2" and Paths.of(it_cfg).moe_dir(0).name == "sd0"
    assert Paths.of(it_cfg).round_dataset(1).name == "base" and Paths.of(it_cfg).round_dataset(3).name == "round3_train"


@pytest.mark.parametrize("override", ["dagger.thresholds=[p90]", "dagger.return_thresholds=[p70]", "dagger.scorers=[vig, flow]",
                                      "moe.seeds=[0, 1]", "sharing.board=/tmp/board", "iterate.start_percentile=120"])
def test_config_refuses(tmp_path, override):
    with pytest.raises(ValueError):
        config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={tmp_path}", override])


def test_adaptive_step():
    ref = np.sort(np.arange(1, 101, dtype=float))  # q(x) = x for x in 1 ... 100
    target = adaptive.percentile_of(ref, ref.mean())
    assert target == 50.0
    p, n_total = adaptive.step(80.0, 0, 10, 70.0, target, ref)  # first attempt: full step, policy steps too high
    assert (p, n_total) == (60.0, 10)
    p, n_total = adaptive.step(p, n_total, 30, 40.0, target, ref)  # 30 of 40 steps, too low: up by 0.75 * 10
    assert (p, n_total) == (67.5, 40)
    assert adaptive.step(p, n_total, 0, None, target, ref) == (67.5, 40)  # no policy steps: unchanged
    assert adaptive.step(99.0, 0, 5, 1.0, target, ref)[0] == 100.0  # clipped
    assert adaptive.threshold_at(ref, 50) == pytest.approx(50.5)


def test_shared_threshold(tmp_path):
    ref = np.arange(1, 101, dtype=float)
    np.savez(tmp_path / "demo_scores.npz", vig=ref[::-1])
    settings = {"adaptive": dict(reference=str(tmp_path / "demo_scores.npz"), target_pct=50.0, start_percentile=80.0)}
    shared = adaptive.SharedThreshold(tmp_path, settings)
    assert shared.current() == (80.0, pytest.approx(np.percentile(ref, 80)))
    e = shared.update([70.0] * 10, 80.0, seed=1, worker=0, success=False)
    assert e["p_after"] == 60.0 and e["n"] == 10 and e["mean_pct"] == 70.0 and not e["success"]
    shared.update([], 60.0, seed=2, worker=1, success=True)  # an attempt without policy steps
    state = json.loads((tmp_path / "threshold_state.json").read_text())
    assert state == dict(p=60.0, total_policy_steps=10, updates=1, attempts=2)
    assert len((tmp_path / "threshold_log.jsonl").read_text().splitlines()) == 2
    assert adaptive.final_percentile(tmp_path) == 60.0


def test_start_percentile_carries_over(it_cfg):
    r1, r2 = collections(it_cfg)[:2]
    assert steps.start_percentile(it_cfg, r1) == 80.0
    with pytest.raises(SystemExit):  # round 1 has not run yet
        steps.start_percentile(it_cfg, r2)
    out = Paths.of(it_cfg).dagger(r1)
    out.mkdir(parents=True)
    (out / "threshold_state.json").write_text(json.dumps(dict(p=88.5, total_policy_steps=1, updates=1, attempts=1)))
    assert steps.start_percentile(it_cfg, r2) == 88.5


def test_graph(it_cfg):
    jobs = {j.name: j for j in plan.build(it_cfg, Path("config.json"))}
    c = [col.id for col in collections(it_cfg)]
    assert jobs["train/moe/round1"].deps == ["convert_planner"]
    assert jobs["round_dataset_2"].deps == [f"analyze_{c[0]}", "convert_planner"]
    assert jobs["train/moe/round2"].deps == ["round_dataset_2"]
    assert set(jobs[f"dagger_prepare_{c[1]}"].deps) == {"thresholds_round2", f"analyze_{c[0]}"}
    assert jobs[f"train/retrain/{c[0]}/seed0"].deps == ["round_dataset_2"]  # the same data round 2's MoE trains on
    assert jobs[f"train/retrain/{c[3]}/seed0"].deps == ["round_dataset_5"]
    assert jobs["round_dataset_5"].deps == [f"analyze_{c[3]}", "round_dataset_4"]
    # round r+1's MoE outranks round r's retraining
    assert jobs["train/moe/round2"].priority < jobs[f"train/retrain/{c[0]}/seed0"].priority
    assert sum(n.startswith(f"dagger_{c[2]}_") for n in jobs) == 4
    assert "--motion" in jobs["collect_planner_000"].args and "fast_maniskill" in jobs["collect_planner_000"].args


def test_runs_resolve_to_round_datasets(it_cfg):
    c = collections(it_cfg)
    assert train.resolve(it_cfg, "moe/round1")[0] == "base"
    assert train.resolve(it_cfg, "moe/round3")[:2] == ("round3_train", Paths.of(it_cfg).moe_dir("round3"))
    assert train.resolve(it_cfg, f"retrain/{c[2].id}/seed0")[0] == "round4_train"


def test_settings_seed_blocks(it_cfg):
    for col in collections(it_cfg)[:2]:  # the adaptive settings need the round's reference
        ref = np.linspace(0.0, 0.05, 101)
        path = Paths.of(it_cfg).demo_scores(col.moe)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vig=ref)
    r1, r2 = collections(it_cfg)[:2]
    out = Paths.of(it_cfg).dagger(r1)
    out.mkdir(parents=True)
    (out / "threshold_state.json").write_text(json.dumps(dict(p=91.0, total_policy_steps=1, updates=1, attempts=1)))
    s1, s2 = steps.dagger_settings(it_cfg, r1), steps.dagger_settings(it_cfg, r2)
    assert (s1["seed_start"], s2["seed_start"]) == (1_000_000, 1_100_000)
    assert s1["take_threshold"] == s1["return_threshold"] == pytest.approx(np.percentile(ref, 80))
    assert s2["adaptive"]["start_percentile"] == 91.0 and s2["target"] == 200
    assert s1["adaptive"]["target_pct"] == adaptive.percentile_of(ref, ref.mean())
    assert s1["policies"].endswith("policies/moe/round1/latest.pt") and s1["expert"] == "fast_maniskill"


def test_training_pieces(tmp_path):
    with h5py.File(tmp_path / "t.h5", "w") as f:
        g = f.create_group("traj_0")
        g["actions"] = np.zeros((60, 8))
        controller = np.zeros(60, np.int8)
        controller[20:25] = 2  # a random move
        controller[50:55] = 2  # one near the end: the 5-step piece after it is too short
        g["dagger/controller"] = controller
        assert training_pieces(g, True, 16) == [(0, 20), (25, 50)]
        assert training_pieces(g, False, 16) == [(0, 60)]


def test_task_numbers_follow_rounds(it_cfg):
    jobs = plan.build(it_cfg, Path("config.json"))
    names = tasks.numbered(it_cfg, jobs)
    first = [n for n in names if "round1" in n]
    assert names[: len(first)] == first and names[-2:] == ["summary", "final"]
    assert names.index("moe/round2") > names.index(f"dagger/{collections(it_cfg)[0].id}")


def test_expert_mode_fast_maniskill():
    from vig_dagger.expert import planner

    try:
        planner.set_expert_mode("fast_maniskill")
        assert (planner.EXPERT_SPEED_TRANSIT, planner.EXPERT_SPEED_INSERT, planner.EXPERT_REFINE_STEPS) == (1.5, 1.0, 0)
        assert planner.EXPERT_PREGRASP_ACC_SCALE == 2.0 and planner.EXPERT_CORRECT_IN_PLACE is False
        assert planner.RANDOM_MOVE_SPEED is None
        assert planner.random_move_speed(planner.StrategyParams.sample(3)) == 1.5
    finally:
        planner.set_expert_mode("fast")
