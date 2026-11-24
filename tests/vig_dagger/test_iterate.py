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
    assert all(n in jobs["final_report"].deps for n in jobs if n.startswith("eval/planner/"))
    names = tasks.numbered(it_cfg, list(jobs.values()))
    assert len(names) == 22 and names.index("planner/round1") < names.index("moe/round2")
    off = config_module.load(REPO / "configs" / "plugcharger.yaml", [f"work_dir={it_cfg['work_dir']}", "iterate.planner_baselines=false"])
    assert not any("planner/" in j.name or "baseline" in j.name for j in plan.build(off, Path("config.json")))


def test_planner_runs_resolve(pb_cfg):
    it_cfg = pb_cfg
    assert train.resolve(it_cfg, "planner/round3/seed0")[:2] == ("planner_750", Paths.of(it_cfg).planner_run(3, 0))
    assert train.resolve(it_cfg, "planner/round1/seed0")[2] == [1]


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
    assert sum(n.startswith("eval/") for n in jobs) == 32 and len(jobs) == 73
    assert tasks.numbered(ref, list(jobs.values())) == ["planner/round1", "planner/round2", "planner/round3",
                                                        "planner/round4", "final"]


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
