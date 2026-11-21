"""This branch: the hand-back threshold sweep, the random-move cut of the retraining data, the expert's strategy check
against the planner demos, and the reference collection."""
import json

import h5py
import numpy as np
import pytest

from vig_dagger import dataset, evaluation, reference, stages, steps, summary
from vig_dagger.config import Collection, Paths, collections
from vig_dagger.envs import POLICY_CAMERAS

THRESHOLDS = reference.ROOT / "sd0_vig_p90_random_move"


def _thresholds(cfg, p90=0.02603707276284695):
    """The seed-0 MoE's VIG thresholds as the reference collection had them."""
    vig = {"p50": 0.0112, "p55": 0.0122, "p60": 0.0131, "p65": 0.0143, "p70": 0.015608374774456021,
           "p75": 0.017319098114967346, "p80": 0.019385353103280066, "p85": 0.021887992322444913, "p90": p90,
           "p95": 0.0324, "p97": 0.0365, "p99": 0.0440}
    path = Paths.of(cfg).thresholds(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(vig=vig, flow={k: v * 1.03 for k, v in vig.items()}, frames=2800, demos=50,
                                    frame_stride=4, seed=0)))


def test_sweep_grid(sweep_cfg):
    cols = collections(sweep_cfg)
    assert [c.id for c in cols] == [f"sd0_vig_p90_r{q}_random_move" for q in (70, 75, 80, 85)]
    for c in cols:
        assert Collection.parse(c.id) == c and c.threshold == "p90" and c.return_spec == f"p{c.id.split('_r')[1][:2]}"
    same = Collection.parse("sd0_vig_p90_random_move")
    assert same.return_threshold is None and same.return_spec == "p90"
    assert Collection(0, "vig", "p90", "random_move", "p90").id == "sd0_vig_p90_random_move"


def test_hand_back_above_take_over_is_refused(write_cfg):
    with pytest.raises(ValueError, match="above the take-over"):
        write_cfg({"dagger": {"thresholds": ["p90"], "return_thresholds": ["p95"]}})
    with pytest.raises(ValueError):
        write_cfg({"dagger": {"thresholds": ["p90"], "return_thresholds": ["p42"]}})


def test_settings_use_both_thresholds(sweep_cfg):
    _thresholds(sweep_cfg)
    s = steps.dagger_settings(sweep_cfg, collections(sweep_cfg)[0])
    assert (s["take_spec"], s["return_spec"]) == ("p90", "p70")
    assert s["take_threshold"] == 0.02603707276284695 and s["return_threshold"] == 0.015608374774456021
    assert (s["take_count"], s["return_count"], s["expert"]) == (2, 16, "demo")


def _traj(f, key, controller, size=16):
    n = len(controller)
    g = f.create_group(key)
    g["actions"] = np.arange(n * 8, dtype=np.float32).reshape(n, 8)
    g["obs/agent/qpos"] = np.zeros((n + 1, 9), np.float32)
    for cam in POLICY_CAMERAS:
        g[f"obs/sensor_data/{cam}/rgb"] = np.full((n + 1, size, size, 3), 100, np.uint8)
    if controller is not None:
        g["dagger/controller"] = np.asarray(controller, np.int8)
    g["success"] = np.r_[np.zeros(n - 1, bool), True]


def test_cut_random_moves(tmp_path):
    src = tmp_path / "worker_00_seg0.h5"
    with h5py.File(src, "w") as f:
        _traj(f, "traj_0", [0] * 20 + [2] * 5 + [1] * 30 + [2] * 3 + [0] * 10)  # pieces [0, 20) [25, 55), [58, 68) short
        _traj(f, "traj_1", [0] * 30)  # no random move: unchanged
    n = dataset.convert([(src, None)], tmp_path / "ds", cut_random_moves=True, min_length=16)
    assert n == 2 and dataset.num_demos(tmp_path / "ds") == 2
    st = dataset.cut_stats(tmp_path / "ds")
    assert st == dict(min_length=16, episodes=2, with_moves=1, moves=2, steps=68, move_steps=8, pieces=3, short_pieces=1,
                      short_steps=10)
    m = json.loads((tmp_path / "ds" / "plug_charger" / "demo_manifest.json").read_text())
    first = m["shards"][0]
    assert [(d["demo_key"], d["length"]) for d in first["demos"]] == [("worker_00_seg0__traj_0_s0-20", 20),
                                                                     ("worker_00_seg0__traj_0_s25-55", 30)]
    side = json.loads((tmp_path / "ds" / "plug_charger" / first["videos"]["side_camera"]["sidecar"]).read_text())
    assert side["frames"] == 68 and [(s["frame_start"], s["frame_count"]) for s in side["segments"]] == [(0, 20), (25, 30)]
    with h5py.File(tmp_path / "ds" / "plug_charger" / first["hdf5"]) as f:
        assert np.array_equal(f["data/worker_00_seg0__traj_0_s25-55/actions"][()],
                              np.arange(68 * 8, dtype=np.float32).reshape(68, 8)[25:55])
    assert [d["demo_key"] for d in m["shards"][1]["demos"]] == ["worker_00_seg0__traj_1"]
    dataset.combine([(tmp_path / "ds", None), (tmp_path / "ds", None)], tmp_path / "both")
    assert dataset.cut_stats(tmp_path / "both")["pieces"] == 6
    top = json.loads((tmp_path / "both" / "dataset_manifest.json").read_text())
    assert top["num_demos"] == 6  # training demos: the pieces


def test_expert_strategy_check(sweep_cfg):
    from vig_dagger.expert.planner import StrategyParams

    raw = Paths.of(sweep_cfg).raw_demos()
    raw.mkdir(parents=True)
    episodes = [dict(episode_id=i, strategy=StrategyParams.sample(1000 + i).to_dict()) for i in range(5)]
    (raw / "part_000.json").write_text(json.dumps(dict(episodes=episodes)))
    assert steps.check_expert_strategies(sweep_cfg) == 5
    episodes[3]["strategy"]["speed_transit"] += 0.01
    (raw / "part_000.json").write_text(json.dumps(dict(episodes=episodes)))
    with pytest.raises(SystemExit, match="1 of 5 differ"):
        steps.check_expert_strategies(sweep_cfg)


def test_reference_matches_and_is_refused(sweep_cfg, monkeypatch):
    assert reference.load(sweep_cfg)[0]["problems"]  # no thresholds yet: this experiment's settings are not known
    _thresholds(sweep_cfg)
    ref = reference.load(sweep_cfg)[0]
    assert ref["id"] == "sd0_vig_p90_random_move" and ref["problems"] == []
    assert len(ref["eval_rows"]) == 200
    _thresholds(sweep_cfg, p90=0.0265)  # another take-over threshold: refused
    assert any("take-over threshold" in p for p in reference.load(sweep_cfg)[0]["problems"])
    _thresholds(sweep_cfg)
    hashes = reference.code_hashes()
    monkeypatch.setattr(reference, "code_hashes", lambda: dict(hashes, **{"expert/planner.py": "0" * 64}))
    assert any("expert/planner.py" in p for p in reference.load(sweep_cfg)[0]["problems"])


def test_summary_and_evaluation_with_reference(sweep_cfg):
    _thresholds(sweep_cfg)
    paths, rng = Paths.of(sweep_cfg), np.random.default_rng(0)
    for k, c in enumerate(collections(sweep_cfg)[:3]):  # the last collection is not analyzed yet
        result = json.loads((THRESHOLDS / "collection.json").read_text())
        result.update(collection=c.id, return_threshold=c.return_spec)
        result["settings"] = dict(result["settings"], collection=c.id, return_spec=c.return_spec, return_threshold=0.016 + k / 1000)
        path = paths.collection_results(c).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    out = summary.summarize(sweep_cfg)
    assert out["collections_analyzed"] == 3 and len(out["references"]) == 1 and not out["references_excluded"]
    ref = out["references"][0]
    assert abs(ref["all_expert"] - 0.3723) < 1e-3 and ref["failed_episodes"] == 120 and ref["failed_expert_frames"] > 100
    assert [s["return_threshold"] for s in out["settings"]] == ["p70", "p75", "p80", "p85"]
    assert (paths.results() / "figures" / "collections.png").stat().st_size > 10_000
    first, n = sweep_cfg["evaluation"]["seed_start"], sweep_cfg["evaluation"]["num_episodes"]

    def episodes(label, rate):
        path = paths.evaluation(label) / "level1" / f"seeds_{first}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for seed in range(first, first + n):
                ok = bool(rng.random() < rate)
                fh.write(json.dumps(dict(seed=seed, factor=1, success=ok, steps=200, furthest_stage="Seated" if ok else "Align",
                                         failed_at=None if ok else "Align", **{f"flag_{k}": False for k in stages.FLAGS})) + "\n")
    episodes("moe/sd0", 0.31)
    for c in collections(sweep_cfg)[:2]:
        episodes(f"retrain/{c.id}/seed0", 0.3)
    ev = evaluation.summarize(sweep_cfg)
    assert len(ev["retrained"]) == 2 and len(ev["references"]) == 1
    r = ev["references"][0]
    assert r["successes"] == 77 and r["episodes"] == 200 and "only_moe" in r
    assert all("sd0_vig_p90_random_move" in x["vs_reference"] for x in ev["retrained"])
    assert (paths.results() / "figures" / "retraining.png").stat().st_size > 10_000


def test_retraining_figure_without_discordant_scenes(sweep_cfg):
    """A policy and the MoE that solve exactly the same scenes have no sign test (p is None): the figure still renders."""
    paths = Paths.of(sweep_cfg)
    first, n = sweep_cfg["evaluation"]["seed_start"], sweep_cfg["evaluation"]["num_episodes"]
    for label in ("moe/sd0", f"retrain/{collections(sweep_cfg)[0].id}/seed0"):
        path = paths.evaluation(label) / "level1" / f"seeds_{first}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(dict(seed=s, factor=1, success=False, steps=400, furthest_stage="Align",
                                                failed_at="Align", **{f"flag_{k}": False for k in stages.FLAGS})) + "\n"
                                for s in range(first, first + n)))
    ev = evaluation.summarize(sweep_cfg)
    assert ev["retrained"][0]["sign_test_p"] is None
    assert (paths.results() / "figures" / "retraining.png").stat().st_size > 10_000
