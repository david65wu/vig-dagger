"""The data behind an interactive page of one DAgger collection (attempt list, attempt viewer, stage tables), in one
file: ``results/dagger_<round>_<scorer>_page.json.gz`` (gzip-compressed JSON, written by ``analysis.py``). The numbers
are the ones behind the DAgger figures (``dagger_figures.py``): the same stage labels, tables and takeover counting.

  round, scorer, experiment     the collection
  stages, flags, signals        stage names; flag key -> label; task-state signal key -> meaning (``stages.SIGNALS``)
  levels                        VIG: the degraded levels' downsampling factors (one gap per level); flow loss: []
  settings                      the collection's settings.json (thresholds, gate counts, target, seeds, step limit)
  demo                          the round's demo-frame scores: n, p5, p25, median, p75, p95, mean, and the shares
                                above the take threshold and below the return threshold
  summary                       attempts, successes, takeovers, attempts with a takeover, returns, steps and expert
                                steps of the attempts used for training, failure reasons, median first-takeover step
  stage_table, flag_table       per group (all, success, failed) and stage / flag (``dagger_figures.stage_stats`` and
                                ``flag_stats``)
  by_stage                      per stage but the last: takeovers per attempt that spent a step there, and the
                                lengths of the takeovers that started there (box quantiles and the values)
  takeovers                     per takeover: seed, success, start, end, length, stage, end_stage, ended (returned /
                                episode_end), trigger_score, position (in its stage: 0 its first step, 1 its last),
                                longest_below (the longest run of scores below the return threshold)
  attempts                      per attempt, in seed order: seed, success, used_for_training, reason, steps, furthest
                                stage, takeovers, returns, expert_steps, first_take (step) and first_stage, flags
                                raised; stage_rle and ctrl_rle (stage and driver per step, as [value, count] runs);
                                score_q (score per step as round(40 log10 score), null when unscored)
  steps                         per attempt (key: seed), per observation: score, gap (one list per VIG level),
                                stage, flags (bit i: the i-th flag), signals {key: values}; per step: ctrl (0 policy,
                                1 expert); takes and returns (the steps where the expert took over / handed back)

Floats keep 4 significant digits (task-state signals: 0.1); NaN (an unscored observation) is null.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

from vig_dagger import dagger_figures as F
from vig_dagger import stages
from vig_dagger.config import Paths

SCORE_SCALE = 40  # score_q = round(SCORE_SCALE * log10(score))


def g4(x) -> float | None:
    x = float(x)
    return float(f"{x:.4g}") if math.isfinite(x) else None


def rle(a) -> list[list[int]]:
    out: list[list[int]] = []
    for v in np.asarray(a).tolist():
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([int(v), 1])
    return out


def clean(obj):
    """JSON-safe: numpy scalars to Python, non-finite floats to null."""
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def _signal(values: np.ndarray) -> list:
    v = np.asarray(values)
    if v.dtype == bool:
        return v.astype(int).tolist()
    return [round(float(x), 1) if math.isfinite(x) else None for x in v]


def build(cfg: dict, round_name: str, scorer: str, settings: dict, episodes: list[dict], takeovers: list[dict],
          per_step: dict[str, list]) -> dict:
    """The page data of one collection, from ``analysis.load_episodes``'s outputs."""
    names = stages.STAGES
    order = sorted(range(len(per_step["seed"])), key=lambda i: per_step["seed"][i])
    rows = [dict(seed=per_step["seed"][i], success=per_step["success"][i], reason=per_step["reason"][i],
                 furthest=per_step["furthest"][i], **{k: per_step[k][i] for k in ("stage", "controller", "score", "flags")})
            for i in order]
    obs = {per_step["seed"][i]: per_step["page"][i] for i in order}
    used = {e["seed"] for e in episodes if e["used_for_training"]}
    detail = {(t["seed"], t["start"]): t for t in F.takeovers(rows, settings)}
    table = sorted(takeovers, key=lambda t: (t["seed"], t["start"]))
    by_seed: dict[int, list[dict]] = {}
    for t in table:
        by_seed.setdefault(t["seed"], []).append(t)

    # takeovers by stage (dagger_takeovers_by_stage.png): per attempt that spent a step in the stage, the takeovers
    # that started there; each takeover's length by its start stage
    by_stage = []
    for k in range(len(names) - 1):
        counts = [sum(t["stage"] == names[k] for t in by_seed.get(a["seed"], [])) for a in rows if (a["stage"] == k).any()]
        lengths = sorted(int(t["length"]) for t in table if t["stage"] == names[k])
        by_stage.append(dict(stage=names[k], counts=F._quantiles(counts), attempts_here=len(counts),
                             with_takeover=sum(c > 0 for c in counts), takeovers=int(sum(counts)),
                             lengths=F._quantiles(lengths), length_values=lengths))

    attempts, steps = [], {}
    for a in rows:
        seed, own, o = a["seed"], by_seed.get(a["seed"], []), obs[a["seed"]]
        q = [None if not math.isfinite(x) or x <= 0 else int(round(SCORE_SCALE * math.log10(x))) for x in a["score"]]
        attempts.append(dict(
            seed=seed, success=a["success"], used_for_training=seed in used, reason=a["reason"], steps=int(len(a["controller"])),
            furthest=names[a["furthest"]], takeovers=len(own), returns=sum(t["ended"] == "returned" for t in own),
            expert_steps=int((a["controller"] == F.EXPERT).sum()), first_take=own[0]["start"] if own else None,
            first_stage=own[0]["stage"] if own else None,
            flags=[n for i, n in enumerate(stages.FLAGS) if (o["flags"] >> i & 1).any()],
            stage_rle=rle(a["stage"]), ctrl_rle=rle(a["controller"]), score_q=q))
        gap = o["gap"]
        steps[str(seed)] = dict(
            score=[g4(x) for x in o["score"]], gap=[[g4(x) for x in col] for col in gap.T] if gap is not None else [],
            stage=[int(x) for x in o["stage"]], flags=[int(x) for x in o["flags"]], ctrl=[int(x) for x in a["controller"]],
            takes=[t["start"] for t in own], returns=[t["end"] for t in own if t["ended"] == "returned"],
            signals={k: _signal(v) for k, v in o["signals"].items()})

    path, demo_values = Paths.of(cfg).demo_scores(round_name), np.zeros(0)
    if path.exists():
        z = np.load(path)
        key = settings.get("scorer_type", scorer)
        if key in z:
            demo_values = np.asarray(z[key], float)
    demo = dict(**F._quantiles(demo_values),
                above_take=float(np.mean(demo_values > settings["take_threshold"])) if len(demo_values) else None,
                below_return=float(np.mean(demo_values < settings["return_threshold"])) if len(demo_values) else None)
    training = [a for a in rows if a["seed"] in used]
    first = [own[0]["start"] for own in by_seed.values()]
    reasons: dict[str, int] = {}
    for a in rows:
        if not a["success"]:
            reasons[a["reason"]] = reasons.get(a["reason"], 0) + 1
    levels = next((o["levels"] for o in obs.values() if o["levels"]), [])
    return clean(dict(
        round=round_name, scorer=scorer, experiment=cfg.get("name"), stages=list(names), flags=dict(stages.FLAGS),
        signals=dict(stages.SIGNALS), levels=levels, score_scale=SCORE_SCALE, settings=settings, demo=demo,
        summary=dict(attempts=len(rows), successes=sum(a["success"] for a in rows), takeovers=len(table),
                     attempts_with_takeover=len(by_seed), returns=sum(t["ended"] == "returned" for t in table),
                     training_attempts=len(training), training_steps=sum(len(a["controller"]) for a in training),
                     training_expert_steps=sum(int((a["controller"] == F.EXPERT).sum()) for a in training),
                     reasons=reasons, first_take_median=float(np.median(first)) if first else None),
        stage_table=F.stage_stats(rows, settings), flag_table=F.flag_stats(rows, settings), by_stage=by_stage,
        takeovers=[dict(seed=t["seed"], success=t["episode_success"], start=t["start"], end=t["end"], length=t["length"],
                        stage=t["stage"], end_stage=t["end_stage"], ended=t["ended"], trigger_score=g4(t["trigger_score"]),
                        position=g4(detail[(t["seed"], t["start"])]["position"]),
                        longest_below=detail[(t["seed"], t["start"])]["longest_below"]) for t in table],
        attempts=attempts, steps=steps))


def write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
