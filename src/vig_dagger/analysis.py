"""DAgger collection metrics (part 3), pooled over steps and as distributions over episodes.

Each attempt (episode) of a collection is summarized from its per-step record (``attempts/*.h5``), where every step
has a driver: the policy, the expert, or (variant ``random_move``) the random move at the start of a takeover:

- ``steps``, ``expert_steps``, ``move_steps``, ``expert_share`` and ``move_share`` (of the attempt's steps),
  ``takeovers`` (policy -> expert switches), ``returns`` (expert -> policy), ``random_moves`` (moves made) and
  ``moves_skipped`` (takeovers where no valid target was found), ``first_take_step`` and the stage it happened in, who
  finished a successful episode (``finished_by``), why a failed one ended (``reason``), the furthest stage reached,
  failure flags;
- score statistics: the share of policy-driven steps whose score is above the take threshold, and of expert-driven
  steps below the return threshold, and the mean scoring time.

Each takeover (the steps from the takeover to the hand-back or the episode's end: the random move, then the
expert): its start and end step, length, random-move and expert steps, the stage at its start and at its end, the
score that triggered it, and how it ended (``returned`` to the policy, or ``episode_end``).

Collection metrics, for four groups of attempts: ``data`` (the first ``dagger.num_demos`` successful episodes to
finish: the collection's DAgger data), ``success``, ``failed`` and ``all``:

- pooled over all steps: expert share and random-move share, takeovers per 100 policy-driven steps;
- distributions over episodes (mean, std, min, p10, p25, median, p75, p90, max) of steps, expert share, random-move
  share, takeovers, returns, random moves, first-takeover step, and over takeovers of the takeover length;
- the share of episodes without any takeover;
- per stage: steps, expert share, takeovers starting there, episodes with a takeover there;
- ``collection``: attempts, successes, the collection success rate (successes / attempts), failure reasons.

Outputs (``results/collections/<collection>``): ``.json`` (all metrics), ``_episodes.csv``, ``_takeovers.csv``,
``_scores.npz`` (the scores on policy- and expert-driven steps), ``_steps.npz`` (every attempt's stage, driver, score
and flags per step), ``_page.json.gz`` (the data of an interactive page of the collection; ``page_data.py``); for the
``random_move`` variant also ``_moves.json`` (every random move, with the third-person camera's parameters) and
``_background.png`` (that camera's first frame of one saved episode), for ``move_figures.py``. With W&B enabled, also
a W&B run ``<name>/dagger/<collection>``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vig_dagger import page_data, stages
from vig_dagger.config import Collection
from vig_dagger.recording import EXPERT, MOVE, POLICY

GROUPS = ("data", "success", "failed", "all")
DISTRIBUTION_KEYS = ("steps", "expert_steps", "move_steps", "expert_share", "move_share", "takeovers", "returns",
                     "random_moves", "first_take_step")


def distribution(values) -> dict[str, float] | None:
    x = np.asarray([v for v in values if v is not None], float)
    if not len(x):
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(n=int(len(x)), mean=float(x.mean()), std=float(x.std()), min=float(x.min()), p10=float(q[0]),
                p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p90=float(q[4]), max=float(x.max()))


def segments(controller: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every takeover: each run of steps the policy did not drive (random move, then expert)."""
    out, start = [], None
    for t, c in enumerate(list(controller) + [POLICY]):
        if c != POLICY and start is None:
            start = t
        elif c == POLICY and start is not None:
            out.append((start, t - start))
            start = None
    return out


def in_group(e: dict, group: str) -> bool:
    return group == "all" or (group == "data" and e["used_for_training"]) or (
        group in ("success", "failed") and e["success"] == (group == "success"))


def load_episodes(out: Path) -> tuple[list[dict], list[dict], dict[str, list[np.ndarray]], dict[str, list]]:
    """Per-episode rows, per-takeover rows, the scores of policy- and expert-driven observations, and every
    attempt's stage, driver, score and flags per step (plus, under ``page``, its per-observation arrays for
    ``page_data.py``)."""
    settings = json.loads((out / "settings.json").read_text())
    take, ret = settings["take_threshold"], settings["return_threshold"]
    summaries = {}
    for path in sorted((out / "episodes").glob("worker_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                summaries[row["seed"]] = row
    used = {s["seed"] for s in sorted((s for s in summaries.values() if s["success"]), key=lambda s: s["finished_at"])
            [: settings["target"]]}
    episodes, takeovers, scores = [], [], {"policy": [], "expert": []}
    per_step: dict[str, list] = {k: [] for k in ("seed", "success", "reason", "furthest", "stage", "controller", "score", "flags", "page")}
    for path in sorted((out / "attempts").glob("worker_*.h5")):
        with h5py.File(path, "r") as f:
            for name in f:
                g = f[name]
                seed = int(g.attrs["seed"])
                if seed not in summaries:  # an attempt interrupted before its summary line
                    continue
                s = summaries[seed]
                state = {k: g[k][:] for k in ("goal_pose", "charger_pose", "tcp_pose", "held", "dist", "angle")}
                controller, score = g["controller"][:], g["score"][:]
                stage, flags = stages.label(state, bool(s["success"]))
                steps = len(controller)
                act_scores = score[:steps]  # the score of the observation each step was taken from
                drove_policy, drove_expert, drove_move = controller == POLICY, controller == EXPERT, controller == MOVE
                pol, exp = act_scores[drove_policy], act_scores[drove_expert]
                pol, exp = pol[np.isfinite(pol)], exp[np.isfinite(exp)]
                scores["policy"].append(pol)
                scores["expert"].append(exp)
                take_steps = list(g["take_steps"][:])
                return_steps = set(int(t) for t in g["return_steps"][:])
                for start, length in segments(controller):
                    end = start + length
                    takeovers.append(dict(
                        seed=seed, episode_success=bool(s["success"]), start=start, end=end, length=length,
                        move_steps=int(drove_move[start:end].sum()), expert_steps=int(drove_expert[start:end].sum()),
                        stage=stages.STAGES[int(stage[start])], end_stage=stages.STAGES[int(stage[min(end, len(stage) - 1)])],
                        trigger_score=float(score[start]), ended="returned" if end in return_steps else "episode_end"))
                per_step["seed"].append(seed)
                per_step["success"].append(bool(s["success"]))
                per_step["reason"].append(str(s["reason"]))
                per_step["furthest"].append(int(stage.max()))
                per_step["stage"].append(stage[:steps].astype(np.uint8))
                per_step["controller"].append(controller.astype(np.uint8))
                per_step["score"].append(act_scores.astype(np.float32))
                bits = sum((flags[k].astype(np.uint8) << i) for i, k in enumerate(stages.FLAGS))
                per_step["flags"].append(bits[:steps])
                per_step["page"].append(dict(score=score, stage=stage, flags=bits, signals=stages.signals(state),
                                             gap=g["vig_gap"][:] if "vig_gap" in g else None,
                                             levels=g["vig_factors"][:].tolist() if "vig_factors" in g else []))
                first = take_steps[0] if take_steps else None
                moves = s.get("random_moves") or []
                episodes.append(dict(
                    seed=seed, success=bool(s["success"]), used_for_training=seed in used, reason=s["reason"],
                    steps=steps, expert_steps=int(drove_expert.sum()), move_steps=int(drove_move.sum()),
                    expert_share=float(drove_expert.mean()) if steps else 0.0,
                    move_share=float(drove_move.mean()) if steps else 0.0,
                    takeovers=len(take_steps), returns=len(return_steps),
                    random_moves=sum(not m.get("skipped") for m in moves), moves_skipped=sum(bool(m.get("skipped")) for m in moves),
                    first_take_step=first, first_take_stage=stages.STAGES[int(stage[first])] if first is not None else None,
                    finished_by=("expert" if controller[-1] == EXPERT else "policy") if s["success"] else None,
                    **stages.summary(stage, flags, bool(s["success"])),
                    policy_above_take=float((pol > take).mean()) if len(pol) else None,
                    expert_below_return=float((exp < ret).mean()) if len(exp) else None,
                    mean_policy_score=float(pol.mean()) if len(pol) else None,
                    score_seconds=s.get("score_seconds"), worker=s["worker"],
                    stage_steps=[int((stage[:steps] == i).sum()) for i in range(len(stages.STAGES))],
                    stage_expert_steps=[int(((stage[:steps] == i) & drove_expert).sum()) for i in range(len(stages.STAGES))]))
    episodes.sort(key=lambda e: e["seed"])
    return episodes, takeovers, scores, per_step


def metrics(episodes: list[dict], takeovers: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in GROUPS:
        eps = [e for e in episodes if in_group(e, group)]
        seeds = {e["seed"] for e in eps}
        segs = [t for t in takeovers if t["seed"] in seeds]
        steps = sum(e["steps"] for e in eps)
        expert = sum(e["expert_steps"] for e in eps)
        move = sum(e["move_steps"] for e in eps)
        m: dict[str, Any] = dict(episodes=len(eps))
        if eps:
            m["pooled"] = dict(steps=steps, expert_steps=expert, move_steps=move,
                               expert_share=expert / steps if steps else 0.0, move_share=move / steps if steps else 0.0,
                               takeovers=sum(e["takeovers"] for e in eps), returns=sum(e["returns"] for e in eps),
                               random_moves=sum(e["random_moves"] for e in eps), moves_skipped=sum(e["moves_skipped"] for e in eps),
                               takeovers_per_100_policy_steps=100 * sum(e["takeovers"] for e in eps) / max(1, steps - expert - move))
            m["per_episode"] = {k: distribution(e[k] for e in eps) for k in DISTRIBUTION_KEYS}
            m["per_takeover"] = dict(length=distribution(t["length"] for t in segs),
                                     move_steps=distribution(t["move_steps"] for t in segs),
                                     returned=float(np.mean([t["ended"] == "returned" for t in segs])) if segs else None)
            m["no_takeover_share"] = float(np.mean([e["takeovers"] == 0 for e in eps]))
            m["stages"] = {}
            for i, name in enumerate(stages.STAGES):
                st = sum(e["stage_steps"][i] for e in eps)
                ex = sum(e["stage_expert_steps"][i] for e in eps)
                m["stages"][name] = dict(steps=st, expert_share=ex / st if st else None,
                                         takeovers=sum(t["stage"] == name for t in segs),
                                         episodes_with_takeover=len({t["seed"] for t in segs if t["stage"] == name}))
            m["first_take_stage"] = {s: sum(e["first_take_stage"] == s for e in eps) for s in stages.STAGES}
            m["furthest_stage"] = {s: sum(e["furthest_stage"] == s for e in eps) for s in stages.STAGES}
            m["flags"] = {k: float(np.mean([e[f"flag_{k}"] for e in eps])) for k in stages.FLAGS}
        if group in ("data", "success") and eps:
            m["finished_by_expert"] = float(np.mean([e["finished_by"] == "expert" for e in eps]))
        if group == "failed" and eps:
            m["reasons"] = {r: sum(e["reason"] == r for e in eps) for r in sorted({e["reason"] for e in eps})}
        out[group] = m
    n = len(episodes)
    out["collection"] = dict(attempts=n, successes=sum(e["success"] for e in episodes),
                             success_rate=sum(e["success"] for e in episodes) / n if n else None,
                             used_for_training=sum(e["used_for_training"] for e in episodes),
                             mean_score_seconds=float(np.mean([e["score_seconds"] for e in episodes if e["score_seconds"]]))
                             if any(e["score_seconds"] for e in episodes) else None)
    return out


MOVE_CAMERA = "third_person_camera"


def write_moves(out: Path, stem: Path, episodes: list[dict]) -> None:
    """Every random move of a random_move collection (``<stem>_moves.json``), and the background for its figures
    (``<stem>_background.png``: the third-person camera's first frame of the scene ``dagger.seed_start`` if it is a
    saved (successful) episode, else of the saved episode with the lowest seed)."""
    import matplotlib.image

    settings = json.loads((out / "settings.json").read_text())
    rows = {}
    for path in sorted((out / "episodes").glob("worker_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["seed"]] = r
    used = {e["seed"] for e in episodes if e["used_for_training"]}
    moves = []
    for seed, r in sorted(rows.items()):
        made = [m for m in r.get("random_moves") or [] if not m.get("skipped")]
        for i, m in enumerate(made):
            moves.append(dict(seed=seed, success=bool(r["success"]), data=seed in used, steps=int(r["steps"]),
                              takeover=i + 1, takeovers=len(made), start_step=int(m["start_step"]), end_step=int(m["end_step"]),
                              progress=round(m["start_step"] / max(1, r["steps"]), 4), held=bool(m["grasped"]),
                              start=m["tcp_start"], end=m["tcp_end"],
                              target=[round(a + b, 5) for a, b in zip(m["tcp_start"], m["offset"])],
                              distance=m["distance"], rot_deg=m["rot_deg"], path_steps=m["path_steps"]))
    saved = sorted((r for r in rows.values() if r["success"] and r.get("file")),
                   key=lambda r: (r["seed"] != settings["seed_start"], r["seed"]))
    camera, background = None, None
    if saved:
        with h5py.File(out / saved[0]["file"], "r") as f:
            t = f[f"traj_{saved[0]['traj']}"]
            sp = t[f"obs/sensor_param/{MOVE_CAMERA}"]
            camera = dict(name=MOVE_CAMERA, extrinsic_cv=sp["extrinsic_cv"][0].tolist(), intrinsic_cv=sp["intrinsic_cv"][0].tolist())
            matplotlib.image.imsave(Path(f"{stem}_background.png"), t[f"obs/sensor_data/{MOVE_CAMERA}/rgb"][0])
        background = saved[0]["seed"]
    Path(f"{stem}_moves.json").write_text(json.dumps(dict(
        radius=settings["random_move"]["radius"], rot_deg=settings["random_move"]["rot_deg"], camera=camera,
        background_seed=background, moves=moves)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = [c for c in rows[0] if not isinstance(rows[0][c], list)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(cfg: dict, collection: Collection, out: Path, stem: Path) -> dict[str, Any]:
    """One collection's metrics, written next to ``stem`` (``results/collections/<collection>``)."""
    from vig_dagger import tracking

    settings = json.loads((out / "settings.json").read_text())
    episodes, takeovers, scores, per_step = load_episodes(out)
    result = dict(collection=collection.id, moe_seed=collection.seed, scorer=collection.scorer,
                  threshold=collection.threshold, variant=collection.variant, settings=settings,
                  metrics=metrics(episodes, takeovers))
    stem.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(Path(f"{stem}_episodes.csv"), episodes)
    _write_csv(Path(f"{stem}_takeovers.csv"), takeovers)
    np.savez_compressed(Path(f"{stem}_scores.npz"),
                        **{who: np.concatenate(v) if v else np.zeros(0) for who, v in scores.items()})
    # Every attempt's stage, driver, score and flags (bit i: stages.FLAGS[i]) per step, concatenated; attempt i is
    # offsets[i]:offsets[i + 1]. Per attempt: seed, success, failure reason, furthest stage.
    lengths = [len(c) for c in per_step["controller"]]
    per_attempt = {k: np.asarray(per_step[k]) for k in ("seed", "success", "reason", "furthest")}
    per_steps = {k: np.concatenate(per_step[k]) if lengths else np.zeros(0) for k in ("stage", "controller", "score", "flags")}
    np.savez_compressed(Path(f"{stem}_steps.npz"), offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                        **per_attempt, **per_steps)
    page_data.write(Path(f"{stem}_page.json.gz"), page_data.build(cfg, collection, settings, episodes, takeovers, per_step))
    if collection.variant == "random_move":
        write_moves(out, stem, episodes)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=1))  # last: its presence marks the analysis done
    m = result["metrics"]
    print(f"[analysis] {collection.id}: {m['collection']}; expert share of the data "
          f"{m['data'].get('pooled', {}).get('expert_share')} (random move {m['data'].get('pooled', {}).get('move_share')}), "
          f"of the failed attempts {m['failed'].get('pooled', {}).get('expert_share')}", flush=True)
    run = tracking.start(cfg, f"dagger/{collection.id}", "dagger-analysis", config=settings,
                         tags=["dagger", f"sd{collection.seed}", collection.scorer, collection.threshold, collection.variant])
    if run is not None:
        run.summary.update(tracking.flatten(m))
        rows = [{k: v for k, v in e.items() if not isinstance(v, list)} for e in episodes]
        run.log({"episodes": tracking.table(rows), "takeovers": tracking.table(takeovers)})
        run.finish()
    return result
