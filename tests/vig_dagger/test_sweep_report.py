"""The threshold-sweep report (sweep_report.py): every number of the threshold analysis and its figures' data in one
Markdown file, and the extra policies evaluated next to the dense ones (evaluation.extra_policies)."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vig_dagger import config as config_module
from vig_dagger import plan, stages, sweep_report, tasks
from vig_dagger.adaptive import percentile_of
from vig_dagger.config import Paths, collections
from vig_dagger.recording import EXPERT, MOVE, POLICY, write_arrays

REPO = Path(__file__).resolve().parents[2]
P, E, M = POLICY, EXPERT, MOVE
# steps 0-10: the policy drives 4 steps, the expert takes over at step 4 (a random move, then 3 expert steps), hands
# back at step 8, the policy drives 3 more. Observation t is scored after step t-1 (observation 0: the reset), so
# observations 3 and 4 triggered the take-over, 5 follows the move (not scored), 6-8 follow expert steps.
CONTROLLER = np.array([P, P, P, P, M, E, E, E, P, P, P])
SCORE = np.array([0.1, 0.2, 0.3, 0.9, 0.95, np.nan, 0.5, 0.4, 0.3, 0.2, 0.25, 0.3])
COUNTED = [0.1, 0.2, 0.3, 0.2, 0.25, 0.3]  # policy steps without the 2 trigger scores
THRESHOLD = 0.8


@pytest.fixture
def sw_cfg(tmp_path):
    return config_module.load(REPO / "configs" / "threshold_planner150.yaml",
                              [f"work_dir={tmp_path / 'work'}", "dagger.thresholds=[p80, p90]",
                               "evaluation.extra_policies={planner_350: {repo: someone/planner, file: planner_350/latest.pt, "
                               "label: planner-only dense policy on 350 demos}}"])


def _state(n, success):
    plug = np.zeros((n, 7), np.float32)
    plug[:, 3] = 1.0
    plug[:, 0] = np.linspace(-0.2, 0.0, n)
    return dict(goal_pose=np.array([0, 0, 0, 1, 0, 0, 0], np.float32), charger_pose=plug, tcp_pose=plug.copy(),
                held=np.ones(n, bool), dist=np.linspace(0.2, 0.0 if success else 0.05, n), angle=np.zeros(n))


def _fake_run(cfg):
    """3 planner demos (their reference scores) and 2 collections of 2 attempts each (one successful, one failed)."""
    paths = Paths.of(cfg)
    raw = paths.raw_demos()
    raw.mkdir(parents=True, exist_ok=True)
    demo, frame, vig = [], [], []
    with h5py.File(raw / "part_000.h5", "w") as f:
        for i in range(3):
            g = f.create_group(f"traj_{i}")
            steps = 40
            success = np.zeros(steps, bool)
            success[30:] = True  # 9 holding steps after the first success
            g["success"], g["actions"] = success, np.zeros((steps, 8), np.float32)
            for k, v in _state(steps + 1, True).items():
                g[f"task_state/{k}"] = v
            for t in range(0, steps, 4):
                demo.append(i), frame.append(t), vig.append(0.01 * (t + i))
    moe = paths.moe_dir(0)
    moe.mkdir(parents=True, exist_ok=True)
    np.savez(paths.demo_scores(0), vig=np.array(vig), demo=np.array(demo), frame=np.array(frame), factors=np.array([1, 2]))
    (moe / "thresholds.json").write_text(json.dumps(dict(vig={"p80": 0.3, "p90": 0.35}, frames=len(vig), demos=3)))
    (moe / "latest.pt").write_bytes(b"weights")
    for col in collections(cfg):
        out = paths.dagger(col)
        for sub in ("episodes", "attempts"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        (out / "settings.json").write_text(json.dumps(dict(take_threshold=THRESHOLD, return_threshold=THRESHOLD,
                                                           take_count=2, return_count=16, target=1)))
        lines = []
        for k, (seed, ok) in enumerate(((1_000_000, True), (1_000_001, False))):
            write_arrays(out / "attempts" / "worker_00.h5", f"seed_{seed}",
                         dict(score=SCORE, controller=CONTROLLER, take_steps=np.array([4]), return_steps=np.array([8]),
                              **_state(len(SCORE), ok)), dict(seed=seed))
            lines.append(json.dumps(dict(seed=seed, success=ok, reason="success_policy" if ok else "time_limit_policy",
                                         steps=len(CONTROLLER), takeovers=1, random_moves=[{}], finished_at=3600.0 * k,
                                         worker=0)))
        (out / "episodes" / "worker_00.jsonl").write_text("\n".join(lines) + "\n")
    first, n = cfg["evaluation"]["seed_start"], cfg["evaluation"]["num_episodes"]
    rng = np.random.default_rng(0)
    for label, rate in (("moe/sd0", 0.05), ("retrain/sd0_vig_p80_random_move/seed0", 0.3),
                        ("retrain/sd0_vig_p90_random_move/seed0", 0.2), ("extra/planner_350", 0.16)):
        path = paths.evaluation(label) / "level1" / f"seeds_{first}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for seed in range(first, first + n):
                ok = bool(rng.random() < rate)
                fh.write(json.dumps(dict(seed=seed, factor=1, success=ok, steps=200 if ok else 400,
                                         furthest_stage="Seated" if ok else "Align", failed_at=None if ok else "Align",
                                         **{f"flag_{k}": False for k in stages.FLAGS})) + "\n")


def test_extra_policy_jobs(sw_cfg):
    jobs = {j.name: j for j in plan.build(sw_cfg, Paths.of(sw_cfg).config())}
    assert "download_extra_planner_350" in jobs
    evals = [n for n in jobs if n.startswith("eval/extra/planner_350/")]
    assert len(evals) == 8 and all(jobs[n].deps == ["download_extra_planner_350"] for n in evals)
    assert jobs["download_extra_planner_350"].task == "eval/extra/planner_350"
    assert "final_report" in jobs and set(evals) <= set(jobs["final_report"].deps)
    names = tasks.numbered(sw_cfg, list(jobs.values()))
    assert names.index("eval/extra/planner_350") == names.index("eval/moe/sd0") + 1


def test_report_round_trip_and_numbers(sw_cfg):
    _fake_run(sw_cfg)
    path = sweep_report.write(sw_cfg)
    data = sweep_report.read(path)
    assert data["version"] == sweep_report.VERSION
    ref_vals = np.sort(np.load(Paths.of(sw_cfg).demo_scores(0))["vig"].astype(float))
    ref = data["reference"]
    assert ref["frames"] == 30 and ref["demos"] == 3
    assert ref["holding_frame_share"] == pytest.approx(6 / 30)  # frames 32 and 36 of each demo come after success
    assert len(ref["percentiles_all"]) == 101 and len(data["bin_edges"]) == 101 + len(sweep_report.OVERFLOW)
    c = {x["threshold"]: x for x in data["collections"]}["p80"]
    assert (c["attempts"], c["successes"]) == (2, 1)
    # the adaptive rule's view: counted policy steps, and the failed attempt's last stretch at the threshold
    q_plain = percentile_of(ref_vals, float(np.mean(COUNTED)))
    q_rule = percentile_of(ref_vals, float(np.mean(COUNTED[:3] + [THRESHOLD] * 3)))
    assert c["adaptive"]["plain"]["equal"] == pytest.approx(q_plain, abs=1e-4)
    assert c["adaptive"]["failed_rule"]["equal"] == pytest.approx((q_plain + q_rule) / 2, abs=1e-4)
    h = c["step_histograms"]
    assert sum(h["policy_counted_success"]) == 6 and sum(h["policy_trigger_success"]) == 2
    assert sum(h["policy_counted_failed_before_last"]) == 3 and sum(h["policy_last_stretch_failed"]) == 3
    assert sum(h["expert_success"]) == sum(h["expert_failed"]) == 3
    assert c["stats"]["policy_counted"]["n"] == 12 and c["stats"]["expert"]["n"] == 6
    assert c["takeovers_per_attempt"]["success"][1] == 1 and c["hours"] == pytest.approx(1.0)
    assert sum(c["expert_segments"]["counts"]) == 2 and c["expert_segments"]["counts"][0] == 2  # 4 steps each
    # the per-scene lines and the paired tests
    pol = data["retraining"]["policies"]
    assert set(pol) == {"moe/sd0", "retrain/sd0_vig_p80_random_move/seed0", "retrain/sd0_vig_p90_random_move/seed0",
                        "extra/planner_350"}
    line = pol["retrain/sd0_vig_p80_random_move/seed0"]["scenes"]
    assert len(line) == 200 and set(line) <= {"0", "1"} and line.count("1") == pol["retrain/sd0_vig_p80_random_move/seed0"]["successes"]
    a = sweep_report.analysis(data)
    assert set(a["dense"]) == {"p80", "p90"} and "p90 vs p80" in a["paired"] and "p80 vs planner_350" in a["paired"]
    assert set(a["rules"]) and all(len(r["values"]) == 2 for r in a["rules"].values())
    text = path.read_text()
    assert "## What the threshold rules say" in text and "planner-only dense policy on 350 demos" in text


def test_figures_from_the_report_alone(sw_cfg, tmp_path):
    _fake_run(sw_cfg)
    path = sweep_report.write(sw_cfg, figures=False)
    copy = tmp_path / "elsewhere" / "REPORT.md"
    copy.parent.mkdir()
    copy.write_text(path.read_text())  # only the file, no work directory
    made = sweep_report.plot(sweep_report.read(copy), tmp_path / "figs")
    assert {m.name for m in made} == {"collections.png", "retraining.png", "distributions.png"}
    assert all(m.stat().st_size > 10_000 for m in made)


def test_download_demos_takes_extra_parts(sw_cfg, monkeypatch):
    """150 demos = raw/planner/part_000 + raw/planner_baselines/part_001-002, every file linked into data/raw/planner."""
    from fnmatch import fnmatch

    import huggingface_hub

    from vig_dagger import hub

    repo_files = ["raw/planner/part_000.h5", "raw/planner/part_000.json", "raw/planner/part_000.done.json"] + [
        f"raw/planner_baselines/part_{k:03d}.{ext}" for k in range(1, 6) for ext in ("h5", "json", "done.json")]

    def fake_download(repo, repo_type, revision, local_dir, allow_patterns):
        for name in repo_files:
            if any(fnmatch(name, p) for p in allow_patterns):
                path = Path(local_dir) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    cfg = dict(sw_cfg, hub=dict(sw_cfg["hub"], dataset_repo="someone/demos"))
    hub.download_demos(cfg)
    raw = Paths.of(cfg).raw_demos()
    assert sorted(p.name for p in raw.glob("part_*.h5")) == ["part_000.h5", "part_001.h5", "part_002.h5"]
    assert (raw / "part_002.done.json").read_text() == "raw/planner_baselines/part_002.done.json"
    jobs = {j.name: j for j in plan.build(dict(cfg, demos=dict(cfg["demos"], source="download")), Paths.of(cfg).config())}
    assert jobs["download_demos"].done()
