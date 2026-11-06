import json

import numpy as np

from vig_dagger import analysis, stages, summary
from vig_dagger.config import Paths, collections


def test_a_takeover_spans_the_random_move_and_the_expert():
    assert analysis.segments(np.array([0, 0, 2, 2, 1, 1, 0, 1, 0, 2])) == [(2, 4), (7, 1), (9, 1)]


def _episode(seed, success, used, controller):
    c = np.asarray(controller)
    n = len(c)
    return dict(seed=seed, success=success, used_for_training=used, reason="success_policy" if success else "time_limit_policy",
                steps=n, expert_steps=int((c == 1).sum()), move_steps=int((c == 2).sum()), expert_share=float((c == 1).mean()),
                move_share=float((c == 2).mean()), takeovers=len(analysis.segments(c)), returns=0, random_moves=1, moves_skipped=0,
                first_take_step=None, first_take_stage=None, finished_by="policy" if success else None,
                furthest_stage=stages.STAGES[-1 if success else 0], **{f"flag_{k}": False for k in stages.FLAGS},
                score_seconds=0.1, stage_steps=[n] + [0] * (len(stages.STAGES) - 1),
                stage_expert_steps=[int((c == 1).sum())] + [0] * (len(stages.STAGES) - 1))


def test_shares_per_group():
    eps = [_episode(1, True, True, [0, 0, 2, 1, 1, 1, 0, 0]), _episode(2, True, False, [0] * 8),
           _episode(3, False, False, [0, 2, 2, 1, 1, 1, 1, 1, 1, 0])]
    m = analysis.metrics(eps, [])
    assert {k: m["collection"][k] for k in ("attempts", "successes", "used_for_training")} == dict(attempts=3, successes=2, used_for_training=1)
    assert m["collection"]["success_rate"] == 2 / 3
    assert m["data"]["pooled"]["expert_share"] == 3 / 8 and m["data"]["pooled"]["move_share"] == 1 / 8
    assert m["success"]["pooled"]["expert_share"] == 3 / 16
    assert m["failed"]["pooled"]["expert_share"] == 6 / 10 and m["failed"]["pooled"]["move_share"] == 2 / 10


def test_summary_and_figure(cfg):
    paths = Paths.of(cfg)
    rng = np.random.default_rng(0)
    for c in collections(cfg):
        if c.seed == 2 and c.scorer == "flow":
            continue  # not analyzed yet
        rate = 0.9 if c.scorer == "vig" else 0.4
        attempts = int(500 / rate)
        expert = {"p90": 0.45, "p95": 0.35, "p99": 0.2}[c.threshold] * (0.4 if c.scorer == "flow" else 1)
        move = 0.1 if c.variant == "random_move" else 0.0
        pooled = lambda e, m: dict(expert_share=e + rng.normal(0, 0.02), move_share=m)
        result = dict(settings=dict(take_threshold=0.03, target=500, max_attempts=10000), metrics=dict(
            collection=dict(attempts=attempts, successes=500, success_rate=500 / attempts),
            data=dict(episodes=500, pooled=pooled(expert, move), per_episode={"expert_share": {"median": expert}}),
            failed=dict(episodes=attempts - 500, pooled=pooled(expert + 0.2, move), per_episode={}, reasons={"time_limit_policy": 3}),
            all=dict(episodes=attempts, pooled=dict(random_moves=10, moves_skipped=1), per_episode={"takeovers": {"median": 2}})))
        path = paths.collection_results(c).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    out = summary.summarize(cfg)
    assert out["collections_analyzed"] == 30 and out["collections_finished"] == 30
    vig = next(s for s in out["settings"] if (s["scorer"], s["variant"], s["threshold"]) == ("vig", "plain", "p90"))
    assert vig["seeds"] == [0, 1, 2] and abs(vig["success_rate"]["mean"] - 500 / 555) < 1e-9
    flow = next(s for s in out["settings"] if (s["scorer"], s["variant"], s["threshold"]) == ("flow", "plain", "p90"))
    assert flow["seeds"] == [0, 1]
    assert (paths.results() / "figures" / "collections.png").stat().st_size > 10_000


def test_move_figures(cfg):
    import matplotlib.image

    from vig_dagger import move_figures
    from vig_dagger.config import Collection

    paths = Paths.of(cfg)
    camera = dict(extrinsic_cv=[[-0.866, -0.5, 0.0, -0.0433], [-0.0356, 0.0617, -0.9975, 0.0281], [0.4988, -0.8638, -0.0713, 0.4481]],
                  intrinsic_cv=[[234.3, 0, 128], [0, 234.3, 128], [0, 0, 1]])
    rng = np.random.default_rng(1)
    stem = paths.collection_results(Collection(1, "vig", "p95", "random_move"))
    stem.parent.mkdir(parents=True, exist_ok=True)
    moves = [dict(seed=s, data=s % 4 != 0, held=bool(s % 2), progress=float(rng.random()),
                  start=[-0.05 + 0.1 * rng.random(), -0.05 + 0.1 * rng.random(), 0.05 + 0.1 * rng.random()],
                  end=[0.0, 0.0, 0.1]) for s in range(40)]
    (stem.parent / f"{stem.name}_moves.json").write_text(json.dumps(dict(radius=0.05, rot_deg=5, camera=camera,
                                                                         background_seed=1000000, moves=moves)))
    matplotlib.image.imsave(stem.parent / f"{stem.name}_background.png", np.full((256, 256, 3), 120, np.uint8))
    made = move_figures.figures(cfg, paths.results() / "figures")
    assert made == [(1, "p95", ["moves/spheres_held_sd1_p95.png", "moves/spheres_progress_sd1_p95.png"])]
    assert (paths.results() / "figures/moves/spheres_held_sd1_p95.png").stat().st_size > 10_000
