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


def test_attempt_weights():
    ref = np.sort(np.arange(1, 101, dtype=float))  # q(x) = x
    # every attempt counts the same and the start counts as one: steps 1/2, 1/3, ... whatever the policy steps
    p, n = adaptive.step(80.0, 0, 400, 100.0, 50.0, ref, updates=0, weight="attempts")
    assert (p, n) == (55.0, 400)  # half way, not all the way to 30
    p, n = adaptive.step(p, n, 5, 20.0, 50.0, ref, updates=1, weight="attempts")
    assert (p, n) == (65.0, 405)  # 5 policy steps move p as much as 400 would (1/3 of 30)
    assert adaptive.step(p, n, 0, None, 50.0, ref, updates=2, weight="attempts") == (65.0, 405)
    # the same two attempts by policy steps: the first moves p fully, the second hardly at all
    p, n = adaptive.step(80.0, 0, 400, 100.0, 50.0, ref)
    assert p == 30.0 and adaptive.step(p, n, 5, 20.0, 50.0, ref)[0] == pytest.approx(30.0 + 5 / 405 * 30)


def test_failed_last_stretch(tmp_path):
    # a failed attempt's scores after its last take-over count at the threshold; nothing else changes
    assert adaptive.counted_scores([1, 2, 3, 4], 2, False, 9.0, True) == ([1.0, 2.0, 9.0, 9.0], 2)
    assert adaptive.counted_scores([1, 2, 3, 4], 0, False, 9.0, True) == ([9.0] * 4, 4)  # never taken over
    assert adaptive.counted_scores([1, 2, 3, 4], 4, False, 9.0, True) == ([1.0, 2.0, 3.0, 4.0], 0)  # expert at the end
    assert adaptive.counted_scores([1, 2, 3, 4], 2, True, 9.0, True) == ([1.0, 2.0, 3.0, 4.0], 0)  # success
    assert adaptive.counted_scores([1, 2, 3, 4], 2, False, 9.0, False) == ([1.0, 2.0, 3.0, 4.0], 0)  # option off
    ref = np.arange(1, 101, dtype=float)
    np.savez(tmp_path / "demo_scores.npz", vig=ref)
    settings = {"adaptive": dict(reference=str(tmp_path / "demo_scores.npz"), target_pct=50.0, start_percentile=80.0,
                                 failed_last_stretch=True)}
    shared = adaptive.SharedThreshold(tmp_path, settings)
    # 10 policy steps at VIG 40 (q = 40, below the target: alone they would raise p), the last 5 of them after the
    # last take-over in a failed attempt -> counted at the threshold p80 = 80.2: mean q 60 -> p goes down by 10
    e = shared.update([40.0] * 10, 80.0, success=False, last_stretch_from=5, seed=1, worker=0)
    assert e["n_last_stretch"] == 5 and e["last_stretch_raw_mean"] == 40.0 and e["mean_pct"] == 60.0
    assert e["p_after"] == pytest.approx(70.0) and e["success"] is False
    e = shared.update([40.0] * 10, 70.0, success=True, last_stretch_from=5, seed=2, worker=0)
    assert e["n_last_stretch"] == 0 and e["mean_pct"] == 40.0  # a successful attempt counts as before


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
    assert s1["adaptive"]["failed_last_stretch"] is True  # on in configs/plugcharger.yaml
    assert s1["adaptive"]["update_weight"] == "attempts" and "/ (updates + 2)" in s1["adaptive"]["rule"]
    it_cfg["iterate"]["failed_last_stretch"] = False  # off: no key, so collections prepared before it compare equal
    it_cfg["iterate"]["update_weight"] = "policy_steps"
    a = steps.dagger_settings(it_cfg, r1)["adaptive"]
    assert "failed_last_stretch" not in a and "update_weight" not in a


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


# ---------------------------------------------------------------- videos and the distribution figure

P, M, E = 0, 2, 1  # recording.POLICY, MOVE, EXPERT
# 12 steps: the policy drives 0-3, obs 3 and 4 are above -> take-over at obs 4; a random move (steps 4-5, obs 5-6 not
# scored); the expert drives 6-8 (obs 7-9 scored), hands back at obs 9; the policy drives 9-11 (obs 10-11 scored).
CONTROLLER = np.array([P, P, P, P, M, M, E, E, E, P, P, P], np.int8)
SCORE = np.array([0.1, 0.2, 0.3, 0.9, 0.95, np.nan, np.nan, 0.6, 0.15, 0.1, 0.25, 0.35, np.nan])


def test_split_scores():
    pol, exp = adaptive.split_scores(SCORE, CONTROLLER, [4], 2, True)
    assert pol.tolist() == [0.1, 0.2, 0.3, 0.25, 0.35] and exp.tolist() == [0.6, 0.15, 0.1]
    pol, _ = adaptive.split_scores(SCORE, CONTROLLER, [4], 2, False)
    assert pol.tolist() == [0.1, 0.2, 0.3, 0.9, 0.95, 0.25, 0.35]


def test_video_jobs_in_graph(it_cfg, tmp_path):
    jobs = {j.name: j for j in plan.build(it_cfg, Path("config.json"))}
    for col in collections(it_cfg):
        v = jobs[f"videos_{col.id}"]
        assert v.deps == [f"analyze_{col.id}"] and v.task == f"dagger/{col.id}"
        assert not any(v.name in j.deps for j in jobs.values())  # nothing waits for the videos
    cfg = config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={tmp_path / 'w'}", "videos.per_kind=0"])
    assert not any(n.startswith("videos_") for n in (j.name for j in plan.build(cfg, Path("config.json"))))


def test_spread():
    from vig_dagger.videos import spread

    assert spread(10, 3) == [0, 4, 9] and spread(2, 5) == [0, 1] and spread(0, 3) == [] and spread(7, 1) == [0]


def _fake_collection(cfg, col, tmp_path):
    """A collection with 2 attempts (one successful, one failed), each saved with tiny camera frames."""
    from vig_dagger.recording import write_arrays

    paths = Paths.of(cfg)
    out = paths.dagger(col)
    for sub in ("episodes", "attempts", "trajectories", "trajectories_failed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ref = np.linspace(0.0, 1.0, 101)
    paths.demo_scores(col.moe).parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths.demo_scores(col.moe), vig=ref)
    (out / "settings.json").write_text(json.dumps(dict(take_threshold=0.8, return_threshold=0.8, take_count=2,
                                                       return_count=16, adaptive=dict(exclude_trigger=True))))
    lines = []
    for k, (seed, ok) in enumerate(((1_000_000, True), (1_000_001, False))):
        n = len(SCORE)
        sub = "trajectories" if ok else "trajectories_failed"
        with h5py.File(out / sub / "worker_00_seg0.h5", "w") as f:
            g = f.create_group("traj_0")
            g["actions"] = np.zeros((n - 1, 8), np.float32)
            for cam in ("third_person_camera", "side_camera", "hand_camera"):
                g[f"obs/sensor_data/{cam}/rgb"] = np.full((n, 32, 32, 3), 40 * k, np.uint8)
            g["dagger/score"], g["dagger/controller"] = SCORE, CONTROLLER
            g["task_state/dist"], g["task_state/held"] = np.linspace(0.1, 0.0, n), np.zeros(n, bool)
        (out / sub / "worker_00_seg0.json").write_text(json.dumps(dict(episodes=[dict(episode_id=0, seed=seed)])))
        write_arrays(out / "attempts" / "worker_00.h5", f"seed_{seed}",
                     dict(score=SCORE, controller=CONTROLLER, take_steps=np.array([4]), return_steps=np.array([9])),
                     dict(seed=seed))
        lines.append(json.dumps(dict(seed=seed, success=ok, reason="success_policy" if ok else "time_limit_policy",
                                     steps=n - 1, takeovers=1, random_moves=[{}], finished_at=float(k), threshold=0.8,
                                     threshold_pct=80.0, worker=0)))
    (out / "episodes" / "worker_00.jsonl").write_text("\n".join(lines) + "\n")
    return out


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg")
def test_render_collection(it_cfg, tmp_path):
    from vig_dagger import videos

    col = collections(it_cfg)[0]
    _fake_collection(it_cfg, col, tmp_path)
    folder = videos.render_collection(it_cfg, col, per_kind=1, workers=1)
    index = json.loads((folder / "index.json").read_text())
    assert [v["file"] for v in index["videos"]] == ["success/attempt0001_seed1000000.mp4", "failed/attempt0002_seed1000001.mp4"]
    assert all((folder / v["file"]).stat().st_size > 1000 for v in index["videos"])
    assert index["attempts"] == 2 and not videos.current(it_cfg, col)  # made with per_kind 1, the config says 20


def test_distribution_figure(it_cfg, tmp_path):
    from vig_dagger import iterate_report

    col = collections(it_cfg)[0]
    _fake_collection(it_cfg, col, tmp_path)
    path = tmp_path / "distributions.png"
    stats = iterate_report.figure_distributions(it_cfg, iterate_report.rounds(it_cfg), path)
    assert path.is_file() and set(stats) == {1}
    s = stats[1]
    assert (s["policy_steps"], s["expert_steps"]) == (10, 6)  # 5 + 3 per attempt
    assert s["policy_mean"] == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.25, 0.35]))
    assert s["policy_above_threshold"] == 0.0 and 0.0 < s["overlap"] < 1.0
    assert s["expert_mean_pct"] is not None


# ---------------------------------------------------------------- planner-only comparisons


@pytest.fixture
def pb_cfg(tmp_path):
    """The experiment with the planner comparison after each round (iterate.planner_baselines: true)."""
    return config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={tmp_path / 'work'}",
                                                                       "iterate.planner_baselines=true"])


def test_planner_sizes_and_parts(it_cfg, pb_cfg):
    from vig_dagger.config import planner_sizes

    assert it_cfg["iterate"]["planner_baselines"] is False and steps.baseline_parts(it_cfg) == []  # runs elsewhere
    it_cfg = pb_cfg
    assert planner_sizes(it_cfg) == [150, 350, 750, 1550]
    parts = steps.baseline_parts(it_cfg)
    assert parts[0] == (1, 50) and parts[-1] == (30, 50) and sum(n for _, n in parts) == 1500  # after the base's part 0
    odd = config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={it_cfg['work_dir']}", "demos.num_demos=60",
                                                                      "iterate.planner_baselines=true"])
    parts = steps.baseline_parts(odd)  # the base fills parts 0-1 (50 + 10): the extra demos start at part 2
    assert parts[0] == (2, 50) and sum(n for _, n in parts) == 1500 and parts[-1] == (31, 50)


def test_planner_jobs(pb_cfg):
    it_cfg = pb_cfg
    jobs = {j.name: j for j in plan.build(it_cfg, Path("config.json"))}
    c = [col.id for col in collections(it_cfg)]
    assert jobs["planner_dataset_1"].deps == ["convert_planner", "collect_baseline_001", "collect_baseline_002"]
    assert jobs["planner_dataset_2"].deps[0] == "planner_dataset_1"
    assert sorted(d for d in jobs["planner_dataset_4"].deps if d.startswith("collect_")) == [
        f"collect_baseline_{k:03d}" for k in range(15, 31)]
    assert jobs["collect_baseline_001"].deps == [] and jobs["collect_baseline_003"].deps == ["collect_baseline_001"]  # 2 lanes
    a = jobs["collect_baseline_005"].args
    assert a[a.index("--start-seed") + 1] == "2005000" and a[a.index("--motion") + 1] == "fast_maniskill"
    for r in (1, 2, 3, 4):  # after the round's own dense policy, then evaluated like it
        t = jobs[f"train/planner/round{r}/seed0"]
        assert t.deps == [f"planner_dataset_{r}", f"train/retrain/{c[r - 1]}/seed0"] and t.task == f"planner/round{r}"
        assert sum(n.startswith(f"eval/planner/round{r}/seed0/") for n in jobs) == 8
        assert t.priority > jobs[f"train/retrain/{c[r - 1]}/seed0"].priority
    t0 = jobs["train/planner/round0/seed0"]  # the dense policy on the base demos alone: where both series start
    assert t0.deps == ["convert_planner"] and t0.task == "planner/round0"
    assert sum(n.startswith("eval/planner/round0/seed0/") for n in jobs) == 8
    assert all(n in jobs["final_report"].deps for n in jobs if n.startswith("eval/planner/"))
    names = tasks.numbered(it_cfg, list(jobs.values()))
    assert len(names) == 23 and names[0] == "planner/round0" and names.index("planner/round1") < names.index("moe/round2")
    off = config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={it_cfg['work_dir']}", "iterate.planner_baselines=false"])
    assert not any("planner/" in j.name or "baseline" in j.name for j in plan.build(off, Path("config.json")))


def test_planner_runs_resolve(pb_cfg):
    it_cfg = pb_cfg
    assert train.resolve(it_cfg, "planner/round3/seed0")[:2] == ("planner_750", Paths.of(it_cfg).planner_run(3, 0))
    assert train.resolve(it_cfg, "planner/round1/seed0")[2] == [1]
    assert train.resolve(it_cfg, "planner/round0/seed0")[:2] == ("base", Paths.of(it_cfg).planner_run(0, 0))


def test_planner_baselines_need_collected_demos(tmp_path):
    with pytest.raises(ValueError, match="planner_baselines"):
        config_module.load(REPO / "configs" / "planner_reference.yaml", [f"work_dir={tmp_path}", "demos.source=download"])
    with pytest.raises(ValueError, match="planner_baselines"):
        config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={tmp_path}", "iterate.planner_baselines=onyl"])


def test_planner_reference_config(tmp_path):
    """configs/planner_reference.yaml: only the planner-only policies, everything else as the DAgger experiment."""
    ref = config_module.load(REPO / "configs" / "planner_reference.yaml", [f"work_dir={tmp_path / 'w'}"])
    exp = config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={tmp_path / 'w'}"])
    for key in ("demos", "training", "moe", "scorers", "thresholds", "dagger", "retrain", "evaluation"):
        assert ref[key] == exp[key], key
    assert {k: v for k, v in ref["iterate"].items() if k not in ("planner_baselines", "baseline_lanes")} == \
        {k: v for k, v in exp["iterate"].items() if k not in ("planner_baselines", "baseline_lanes")}
    assert ref["iterate"]["planner_baselines"] == "only"
    jobs = {j.name: j for j in plan.build(ref, Path("config.json"))}
    kinds = {n.split("/")[0].rstrip("_0123456789") for n in jobs}
    assert kinds == {"collect_planner", "convert_planner", "collect_baseline", "planner_dataset", "train", "eval",
                     "final_report"}
    assert all(jobs[f"train/planner/round{r}/seed0"].deps == [f"planner_dataset_{r}"] for r in (1, 2, 3, 4))
    assert jobs["collect_baseline_009"].deps == ["collect_baseline_001"] and jobs["collect_baseline_008"].deps == []
    assert jobs["train/planner/round0/seed0"].deps == ["convert_planner"]
    assert sum(n.startswith("eval/") for n in jobs) == 40 and len(jobs) == 82
    assert tasks.numbered(ref, list(jobs.values())) == ["planner/round0", "planner/round1", "planner/round2",
                                                        "planner/round3", "planner/round4", "final"]


def test_check_planner_part(it_cfg, tmp_path):
    done = tmp_path / "part_003.done.json"
    done.write_text(json.dumps(dict(part=3, demos=50, first_seed=2_003_000, strategy_seed=4, motion="fast_maniskill")))
    steps.check_planner_part(it_cfg, done, 3)
    done.write_text(json.dumps(dict(part=3, demos=50, first_seed=2_003_000, strategy_seed=4, motion="demo")))
    with pytest.raises(SystemExit, match="motion"):
        steps.check_planner_part(it_cfg, done, 3)
    done.write_text(json.dumps(dict(part=3, demos=50, imported_from="/x.h5")))  # imported: nothing to check
    steps.check_planner_part(it_cfg, done, 3)


def test_baseline_sources(pb_cfg):
    it_cfg = pb_cfg
    raw = Paths.of(it_cfg).raw_baselines()
    raw.mkdir(parents=True)
    for k in (1, 2, 3):
        with h5py.File(raw / f"part_{k:03d}.h5", "w") as f:
            for i in range(50):
                f.create_group(f"traj_{i}")
    src = steps.baseline_sources(it_cfg, 0, 100)  # round 1: the first 100 extra demos
    assert [(p.name, len(i)) for p, i in src] == [("part_001.h5", 50), ("part_002.h5", 50)]
    src = steps.baseline_sources(it_cfg, 100, 120)
    assert [(p.name, i) for p, i in src] == [("part_003.h5", list(range(20)))]


def _fake_results(cfg, col, k):
    """A collection's analysis result (results/collections/<c>.json) with made-up numbers."""
    path = Paths.of(cfg).collection_results(col).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    pooled = lambda e, m: dict(expert_share=e, move_share=m, expert_steps=1000 * e, move_steps=1000 * m)  # noqa: E731
    metrics = dict(collection=dict(attempts=100 + 20 * k, successes=100, success_rate=100 / (100 + 20 * k)),
                   data=dict(episodes=100, pooled=pooled(0.4 + 0.05 * k, 0.12),
                             per_episode=dict(takeovers=dict(mean=2.0 + 0.2 * k, std=0.7))),
                   all=dict(pooled=pooled(0.42, 0.11), per_episode={}),
                   failed=dict(episodes=20 * k, pooled=pooled(0.3, 0.1), reasons={}))
    path.write_text(json.dumps(dict(metrics=metrics, settings=dict(target=100, max_attempts=2000, take_threshold=0.03))))


def test_collections_and_retraining_figures(it_cfg, tmp_path):
    from vig_dagger import iterate_report

    for k, col in enumerate(collections(it_cfg)[:3]):
        _fake_results(it_cfg, col, k)
    rows = [dict(round=r, target=t, moe_training_demos=49 + 100 * (r - 1), start_pct=80.0 if r == 1 else 88.0,
                 final_pct=88.0 + r / 10, attempts=150, success_rate=0.7) for r, t in zip((1, 2, 3, 4), (100, 200, 400, 800))]
    rows[0].update(dense_successes=7, dense_episodes=200, moe_successes=0, moe_episodes=200, planner_successes=7,
                   planner_episodes=200, planner_demos=150, dense_vs_moe=dict(only_dense=7, only_moe=0, p=0.016))
    rows[1].update(dense_successes=40, dense_episodes=200, dense_vs_previous=dict(only_this=35, only_previous=2, p=1e-8))
    entries = iterate_report._collection_entries(it_cfg, rows)
    assert [e["round"] for e in entries] == [1, 2, 3] and entries[2]["takeovers"] == pytest.approx(2.4)
    assert entries[0]["info"]["start_pct"] == 80.0
    assert iterate_report.figure_collections(it_cfg, rows, tmp_path / "collections.png")
    assert iterate_report.figure_retraining(it_cfg, rows, tmp_path / "retraining.png",
                                            dict(demos=50, successes=1, episodes=200, success_rate=0.005))
    assert iterate_report.figure_rounds(rows, tmp_path / "rounds.png", dict(demos=50, successes=1, episodes=200))
    assert (tmp_path / "collections.png").is_file() and (tmp_path / "retraining.png").is_file()
    assert not iterate_report.figure_retraining(it_cfg, [dict(round=1, target=100)], tmp_path / "none.png")


def test_round_move_figures(it_cfg):
    import matplotlib.image

    from vig_dagger import move_figures

    paths = Paths.of(it_cfg)
    camera = dict(extrinsic_cv=[[-0.866, -0.5, 0.0, -0.0433], [-0.0356, 0.0617, -0.9975, 0.0281],
                                [0.4988, -0.8638, -0.0713, 0.4481]],
                  intrinsic_cv=[[234.3, 0, 128], [0, 234.3, 128], [0, 0, 1]])
    rng = np.random.default_rng(1)
    assert move_figures.round_figures(it_cfg, paths.results() / "figures") == []  # nothing analyzed yet
    for col in collections(it_cfg)[:2]:  # rounds 1 and 2 analyzed, 3 and 4 not yet
        stem = paths.collection_results(col)
        stem.parent.mkdir(parents=True, exist_ok=True)
        moves = [dict(seed=s, data=True, held=bool(s % 2), progress=float(rng.random()),
                      start=[-0.05 + 0.1 * rng.random(), -0.05 + 0.1 * rng.random(), 0.05 + 0.1 * rng.random()],
                      end=[0.0, 0.0, 0.1]) for s in range(30)]
        Path(f"{stem}_moves.json").write_text(json.dumps(dict(radius=0.05, rot_deg=5, camera=camera,
                                                              background_seed=1000000, moves=moves)))
        matplotlib.image.imsave(Path(f"{stem}_background.png"), np.full((256, 256, 3), 120, np.uint8))
    made = move_figures.round_figures(it_cfg, paths.results() / "figures")
    assert made == ["moves/spheres_held_rounds.png", "moves/spheres_progress_rounds.png"]
    assert (paths.results() / "figures" / made[0]).stat().st_size > 10_000
