import json

import numpy as np

from vig_dagger import evaluation, stages
from vig_dagger.config import Paths, collections


def _episodes(path, first, n, rate, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for seed in range(first, first + n):
            ok = bool(rng.random() < rate)
            fh.write(json.dumps(dict(seed=seed, factor=1, success=ok, steps=200 if ok else 400, furthest_stage="Seated" if ok else "Align",
                                     failed_at=None if ok else "Align", **{f"flag_{k}": False for k in stages.FLAGS})) + "\n")


def test_sign_test():
    assert evaluation.sign_test(0, 0) is None
    assert evaluation.sign_test(5, 5) == 1.0
    assert abs(evaluation.sign_test(10, 0) - 2 / 2**10) < 1e-12


def test_summary_and_figure(cfg):
    paths, rng = Paths.of(cfg), np.random.default_rng(0)
    first, n = cfg["evaluation"]["seed_start"], cfg["evaluation"]["num_episodes"]
    for seed in cfg["moe"]["seeds"]:
        _episodes(paths.evaluation(f"moe/sd{seed}") / "level1" / f"seeds_{first}.jsonl", first, n, 0.34, rng)
    for c in collections(cfg):
        if c.seed == 2 and c.variant == "random_move":
            continue  # not evaluated yet
        rate = {"p90": 0.6, "p95": 0.5, "p99": 0.4}[c.threshold] - (0.1 if c.scorer == "flow" else 0)
        _episodes(paths.evaluation(f"retrain/{c.id}/seed0") / "level1" / f"seeds_{first}.jsonl", first, n, rate, rng)
    out = evaluation.summarize(cfg)
    assert len(out["retrained"]) == 30 and set(out["moes"]) == {"0", "1", "2"}
    vig = next(s for s in out["settings"] if (s["scorer"], s["variant"], s["threshold"]) == ("vig", "plain", "p90"))
    assert vig["moe_seeds"] == [0, 1, 2] and 0.5 < vig["success_mean"] < 0.7 and vig["sign_test_p"] < 0.01
    rm = next(s for s in out["settings"] if (s["scorer"], s["variant"], s["threshold"]) == ("vig", "random_move", "p90"))
    assert rm["moe_seeds"] == [0, 1]
    assert (paths.results() / "figures" / "eval_success.png").stat().st_size > 10_000
