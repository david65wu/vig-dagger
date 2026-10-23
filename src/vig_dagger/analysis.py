"""DAgger collection metrics (part 3), pooled over frames and as distributions over episodes.

Each attempt (episode) of a scorer's collection is summarized from its per-step record (``attempts/*.h5``):

- ``steps``, ``expert_steps``, ``expert_share`` (expert steps / steps), ``takeovers`` (policy -> expert switches),
  ``returns`` (expert -> policy), ``first_take_step`` and the stage it happened in, who finished a successful
  episode (``finished_by``), why a failed one ended (``reason``), the furthest stage reached, failure flags;
- score statistics: the share of policy-driven steps whose score is above the take threshold, and of expert-driven
  steps below the return threshold, and the mean scoring time.

Each takeover (an expert segment): its start step, length, the stage at its start, the score that triggered it,
and how it ended (``returned`` to the policy, or ``episode_end``).

Collection metrics, for successful, failed and all attempts:

- counts and the collection success rate (successes / attempts), failure reasons;
- pooled over all steps: expert share, takeovers per 100 policy-driven steps;
- distributions over episodes (mean, std, min, p10, p25, median, p75, p90, max) of steps, expert share, takeovers,
  returns, first-takeover step, and over takeovers of the expert-segment length;
- the share of episodes without any takeover;
- per stage: steps, expert share, takeovers starting there, episodes with a takeover there.

Outputs: ``results/dagger_<round>_<scorer>.json`` (all metrics), ``results/dagger_<round>_<scorer>_episodes.csv``,
``_takeovers.csv``, ``_scores.npz`` (the scores on policy- and expert-driven steps) and
``results/figures/dagger_<round>_<scorer>_timelines.png``; ``report.py`` turns them into tables and figures. With
W&B enabled, also a W&B run ``<name>/dagger/<round>/<scorer>`` with the scalars, the two tables,
histograms and episode timelines; ``<name>/dagger/<round>/compare`` puts the scorers side by side.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vig_dagger import stages
from vig_dagger.recording import EXPERT, POLICY

GROUPS = ("success", "failed", "all")
DISTRIBUTION_KEYS = ("steps", "expert_steps", "expert_share", "takeovers", "returns", "first_take_step")


def distribution(values) -> dict[str, float] | None:
    x = np.asarray([v for v in values if v is not None], float)
    if not len(x):
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return dict(n=int(len(x)), mean=float(x.mean()), std=float(x.std()), min=float(x.min()), p10=float(q[0]),
                p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p90=float(q[4]), max=float(x.max()))


def segments(controller: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of expert-driven steps."""
    out, start = [], None
    for t, c in enumerate(list(controller) + [POLICY]):
        if c == EXPERT and start is None:
            start = t
        elif c != EXPERT and start is not None:
            out.append((start, t - start))
            start = None
    return out


def load_episodes(out: Path) -> tuple[list[dict], list[dict], dict[str, list[np.ndarray]]]:
    """Per-episode rows, per-takeover rows, and the scores of policy- and expert-driven observations."""
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
                drove_policy, drove_expert = controller == POLICY, controller == EXPERT
                pol, exp = act_scores[drove_policy], act_scores[drove_expert]
                pol, exp = pol[np.isfinite(pol)], exp[np.isfinite(exp)]
                scores["policy"].append(pol)
                scores["expert"].append(exp)
                take_steps = list(g["take_steps"][:])
                return_steps = set(int(t) for t in g["return_steps"][:])
                for start, length in segments(controller):
                    end = start + length
                    takeovers.append(dict(
                        seed=seed, episode_success=bool(s["success"]), start=start, length=length,
                        stage=stages.STAGES[int(stage[start])], trigger_score=float(score[start]),
                        ended="returned" if end in return_steps else "episode_end"))
                first = take_steps[0] if take_steps else None
                episodes.append(dict(
                    seed=seed, success=bool(s["success"]), used_for_training=seed in used, reason=s["reason"],
                    steps=steps, expert_steps=int(drove_expert.sum()), expert_share=float(drove_expert.mean()) if steps else 0.0,
                    takeovers=len(take_steps), returns=len(return_steps), first_take_step=first,
                    first_take_stage=stages.STAGES[int(stage[first])] if first is not None else None,
                    finished_by=("expert" if controller[-1] == EXPERT else "policy") if s["success"] else None,
                    **stages.summary(stage, flags, bool(s["success"])),
                    policy_above_take=float((pol > take).mean()) if len(pol) else None,
                    expert_below_return=float((exp < ret).mean()) if len(exp) else None,
                    mean_policy_score=float(pol.mean()) if len(pol) else None,
                    score_seconds=s.get("score_seconds"), worker=s["worker"],
                    stage_steps=[int((stage[:steps] == i).sum()) for i in range(len(stages.STAGES))],
                    stage_expert_steps=[int(((stage[:steps] == i) & drove_expert).sum()) for i in range(len(stages.STAGES))]))
    episodes.sort(key=lambda e: e["seed"])
    return episodes, takeovers, scores


def metrics(episodes: list[dict], takeovers: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in GROUPS:
        eps = [e for e in episodes if group == "all" or e["success"] == (group == "success")]
        seeds = {e["seed"] for e in eps}
        segs = [t for t in takeovers if t["seed"] in seeds]
        steps = sum(e["steps"] for e in eps)
        expert = sum(e["expert_steps"] for e in eps)
        m: dict[str, Any] = dict(episodes=len(eps))
        if eps:
            m["pooled"] = dict(steps=steps, expert_steps=expert, expert_share=expert / steps if steps else 0.0,
                               takeovers=sum(e["takeovers"] for e in eps), returns=sum(e["returns"] for e in eps),
                               takeovers_per_100_policy_steps=100 * sum(e["takeovers"] for e in eps) / max(1, steps - expert))
            m["per_episode"] = {k: distribution(e[k] for e in eps) for k in DISTRIBUTION_KEYS}
            m["per_takeover"] = dict(length=distribution(t["length"] for t in segs),
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
        if group == "success" and eps:
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = [c for c in rows[0] if not isinstance(rows[0][c], list)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def timeline_figure(out: Path, seeds: list[int], take: float, ret: float, title: str):
    """Score over the episode with both thresholds and the expert-driven steps shaded, one panel per seed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(seeds), 1, figsize=(10, 2.2 * len(seeds)), squeeze=False)
    for ax, seed in zip(axes[:, 0], seeds):
        for path in (out / "attempts").glob("worker_*.h5"):
            with h5py.File(path, "r") as f:
                if f"seed_{seed}" not in f:
                    continue
                g = f[f"seed_{seed}"]
                score, controller = g["score"][:], g["controller"][:]
                ax.plot(score, lw=1, color="#3b6ea5")
                for start, length in segments(controller):
                    ax.axvspan(start, start + length, color="#e8a33d", alpha=0.25, lw=0)
                ax.axhline(take, color="#c0392b", lw=0.8, ls="--")
                ax.axhline(ret, color="#27ae60", lw=0.8, ls="--")
                ax.set_title(f"seed {seed}: {'success' if g.attrs['success'] else 'failed'} ({g.attrs['reason']})", fontsize=9)
                ax.set_ylabel("score")
    axes[-1, 0].set_xlabel("step (shaded: expert driving; red: take threshold; green: return threshold)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def analyze(cfg: dict, round_name: str, scorer: str, out: Path, results: Path) -> dict[str, Any]:
    from vig_dagger import tracking

    settings = json.loads((out / "settings.json").read_text())
    episodes, takeovers, scores = load_episodes(out)
    result = dict(round=round_name, scorer=scorer, settings=settings, metrics=metrics(episodes, takeovers))
    results.mkdir(parents=True, exist_ok=True)
    stem = results / f"dagger_{round_name}_{scorer}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=1))
    _write_csv(Path(f"{stem}_episodes.csv"), episodes)
    _write_csv(Path(f"{stem}_takeovers.csv"), takeovers)
    print(f"[analysis] {round_name}/{scorer}: {result['metrics']['collection']}; pooled expert share "
          f"{result['metrics']['all'].get('pooled', {}).get('expert_share')}", flush=True)
    # The scores on policy- and expert-driven steps (for the report's histograms), and six score timelines.
    np.savez_compressed(Path(f"{stem}_scores.npz"),
                        **{who: np.concatenate(v) if v else np.zeros(0) for who, v in scores.items()})
    picks = [e["seed"] for e in episodes if e["success"]][:3] + [e["seed"] for e in episodes if not e["success"]][:3]
    timelines = None
    if picks:
        import matplotlib.pyplot as plt

        timelines = timeline_figure(out, picks, settings["take_threshold"], settings["return_threshold"],
                                    f"{round_name} round, {scorer} DAgger: score per step (shaded: expert driving)")
        (results / "figures").mkdir(parents=True, exist_ok=True)
        timelines.savefig(results / "figures" / f"dagger_{round_name}_{scorer}_timelines.png", dpi=90)
    run = tracking.start(cfg, f"dagger/{round_name}/{scorer}", "dagger-analysis", config=settings, tags=["dagger", round_name, scorer])
    if run is not None:
        import wandb

        run.summary.update(tracking.flatten(result["metrics"]))
        rows = [{k: v for k, v in e.items() if not isinstance(v, list)} for e in episodes]
        log = {"episodes": tracking.table(rows), "takeovers": tracking.table(takeovers)}
        for group in GROUPS:
            eps = [e for e in episodes if group == "all" or e["success"] == (group == "success")]
            for key in ("expert_share", "takeovers", "returns", "steps", "first_take_step"):
                values = [e[key] for e in eps if e[key] is not None]
                if values:
                    log[f"hist/{group}/{key}"] = wandb.Histogram(values)
        lengths = [t["length"] for t in takeovers]
        if lengths:
            log["hist/all/takeover_length"] = wandb.Histogram(lengths)
        for who in ("policy", "expert"):
            values = np.concatenate(scores[who]) if scores[who] else np.array([])
            if len(values):
                log[f"hist/score_{who}_driven"] = wandb.Histogram(values)
        stage_rows = [dict(stage=s, **v) for s, v in result["metrics"]["all"].get("stages", {}).items()]
        if stage_rows:
            log["stages"] = tracking.table(stage_rows)
        if timelines is not None:
            log["timelines"] = wandb.Image(timelines)
        run.log(log)
        run.finish()
    if timelines is not None:
        import matplotlib.pyplot as plt

        plt.close(timelines)
    return result


def compare(cfg: dict, round_name: str, results: dict[str, dict]) -> None:
    """One W&B run with the scorers side by side: a table of the main collection metrics and their distributions."""
    from vig_dagger import tracking

    rows = []
    for scorer, r in results.items():
        m = r["metrics"]
        for group in GROUPS:
            g = m[group]
            if not g.get("episodes"):
                continue
            pe = g["per_episode"]
            rows.append(dict(
                scorer=scorer, group=group, episodes=g["episodes"], success_rate=m["collection"]["success_rate"],
                pooled_expert_share=g["pooled"]["expert_share"],
                takeovers_per_100_policy_steps=g["pooled"]["takeovers_per_100_policy_steps"],
                median_expert_share=pe["expert_share"]["median"], mean_expert_share=pe["expert_share"]["mean"],
                median_takeovers=pe["takeovers"]["median"], mean_takeovers=pe["takeovers"]["mean"],
                mean_steps=pe["steps"]["mean"], no_takeover_share=g["no_takeover_share"],
                take_threshold=r["settings"]["take_threshold"], return_threshold=r["settings"]["return_threshold"]))
    (Path(cfg["work_dir"]) / "results" / f"dagger_{round_name}_compare.json").write_text(json.dumps(rows, indent=1))
    run = tracking.start(cfg, f"dagger/{round_name}/compare", "dagger-analysis", tags=["dagger", round_name, "compare"])
    if run is not None:
        import wandb

        table = tracking.table(rows)
        log = {"compare": table}
        for metric in ("pooled_expert_share", "median_expert_share", "mean_takeovers", "success_rate"):
            sub = [[f"{r['scorer']} ({r['group']})", r[metric]] for r in rows if r[metric] is not None]
            log[f"bar/{metric}"] = wandb.plot.bar(wandb.Table(data=sub, columns=["run", metric]), "run", metric, title=metric)
        run.log(log)
        run.finish()
