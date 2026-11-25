"""The report of an iteration-mode experiment (``iterate.rounds``): ``results/REPORT.md``, ``results/iterate_summary.json``
and the figures ``results/figures/collections.png`` (per round's collection, as bars: success rate per attempt, expert
and random-move share of the DAgger data, of all attempts and of the failed attempts, takeovers per kept episode),
``results/figures/retraining.png`` (per round, as bars with Wilson intervals: the dense policy retrained after the round,
the dense policy on as many planner demos when evaluated, and the round's MoE; paired tests in the footnote),
``results/figures/moves/spheres_{held,progress}_rounds.png`` (every random move of each round's DAgger data on the
third-person camera, one panel per round; ``move_figures.py``), ``results/figures/thresholds.png`` (the adaptive threshold attempt by attempt, round after round),
``results/figures/rounds.png`` (per round: the collection's success rate and expert share, the dense policy retrained
after the round, the dense policy on as many planner demos with ``iterate.planner_baselines``, and that round's MoE on
the evaluation scenes) and ``results/figures/distributions.png`` (per round:
VIG histograms of the MoE's training frames and of its collection's policy steps and policy + expert steps, with their
overlap; ``iterate_summary.json`` has the numbers under each round's ``distribution``).

Per round: the MoE's training demos and reference (frames, mean VIG M, its percentile q(M): the target), where the
threshold started and ended, the collection (attempts, successes, expert and random-move share of its DAgger data), the
dense policy retrained on the base demos plus every round's DAgger data so far, and paired tests on the evaluation
scenes: that policy against the previous round's and against the round's MoE.
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import numpy as np

from vig_dagger.adaptive import final_percentile
from vig_dagger.config import Paths, collections, planner_sizes, target_of
from vig_dagger.evaluation import _paired, load, sign_test, wilson

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
ROUND_COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#8e44ad", "#c76fa5", "#6b7280"]


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _log(out: Path) -> list[dict]:
    path = out / "threshold_log.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["time"])


def rounds(cfg: dict) -> list[dict]:
    paths = Paths.of(cfg)
    n_eval = int(cfg["evaluation"]["num_episodes"])
    evals = load(paths.root / "eval") if (paths.root / "eval").is_dir() else {}
    complete = {k: v for k, v in evals.items() if len(v) >= n_eval}
    # the dense policy on the base demos alone (planner round 0): round 1's dense policy is compared with it
    out, previous_dense = [], complete.get(f"planner/round0/seed{int(cfg['retrain']['seeds'][0])}")
    for col in collections(cfg):
        r, out_dir = col.round, paths.dagger(col)
        thr = _json(paths.thresholds(col.moe)) or {}
        run = _json(paths.moe_dir(col.moe) / "run_config.json") or {}
        settings = _json(out_dir / "settings.json") or {}
        metrics = (_json(paths.collection_results(col).with_suffix(".json")) or {}).get("metrics", {})
        log = _log(out_dir)
        coll = metrics.get("collection", {})
        data = metrics.get("data", {}).get("pooled", {})
        row = dict(round=r, collection=col.id, target=target_of(cfg, col),
                   moe_training_demos=(run.get("_info") or {}).get("training_demos"),
                   reference_frames=thr.get("frames"), reference_demos=thr.get("demos"),
                   reference_mean=thr.get("vig_mean"), target_pct=thr.get("vig_mean_percentile"),
                   start_pct=(settings.get("adaptive") or {}).get("start_percentile"), final_pct=final_percentile(out_dir),
                   updates=len(log), attempts=coll.get("attempts"), successes=coll.get("successes"),
                   success_rate=coll.get("success_rate"), expert_share=data.get("expert_share"),
                   move_share=data.get("move_share"))
        if log:
            half = [e["used_p"] for e in log[len(log) // 2:]]
            row.update(second_half_median_pct=float(np.median(half)), second_half_p10=float(np.percentile(half, 10)),
                       second_half_p90=float(np.percentile(half, 90)),
                       final_threshold=log[-1]["threshold_after"])
        seed0 = int(cfg['retrain']['seeds'][0])
        dense = complete.get(f"retrain/{col.id}/seed{seed0}")
        moe = complete.get(f"moe/{col.moe}")
        # the planner-only policy: trained here (iterate.planner_baselines) or on another machine, its evaluation
        # copied into eval/planner/
        planner = complete.get(f"planner/round{r}/seed{seed0}")
        row["planner_demos"] = planner_sizes(cfg)[r - 1]
        for name, rows_ in (("dense", dense), ("moe", moe), ("planner", planner)):
            if rows_ is not None:
                k = int(sum(e["success"] for e in rows_))
                row[f"{name}_successes"], row[f"{name}_episodes"] = k, len(rows_)
                row[f"{name}_success_rate"] = k / len(rows_)
        if dense is not None and moe is not None:
            w, lo = _paired(dense, moe)
            row["dense_vs_moe"] = dict(only_dense=w, only_moe=lo, p=sign_test(w, lo))
        if dense is not None and planner is not None:
            w, lo = _paired(dense, planner)
            row["dense_vs_planner"] = dict(only_dense=w, only_planner=lo, p=sign_test(w, lo))
        if dense is not None and previous_dense is not None:
            w, lo = _paired(dense, previous_dense)
            row["dense_vs_previous"] = dict(only_this=w, only_previous=lo, p=sign_test(w, lo))
        previous_dense = dense
        out.append(row)
    return out


def base_dense(cfg: dict) -> dict | None:
    """The dense policy on the base demos alone (``planner/round0/seed<s>``): successes and episodes, once its
    evaluation is complete."""
    paths = Paths.of(cfg)
    if not (paths.root / "eval").is_dir():
        return None
    label = f"planner/round0/seed{int(cfg['retrain']['seeds'][0])}"
    rows = load(paths.root / "eval").get(label)
    if not rows or len(rows) < int(cfg["evaluation"]["num_episodes"]):
        return None
    k = int(sum(e["success"] for e in rows))
    return dict(demos=int(cfg["demos"]["num_demos"]), successes=k, episodes=len(rows), success_rate=k / len(rows))


def _pct(x, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f} %"


def _p(x) -> str:
    return "–" if x is None else f"p{x:.1f}"


def _test(t: dict | None, a: str, b: str) -> str:
    if not t:
        return "–"
    p = t["p"]
    return f"{t[a]} / {t[b]}, p {p:.3g}" if p is not None else f"{t[a]} / {t[b]}"


def figure_thresholds(cfg: dict, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = Paths.of(cfg)
    logs = [(col, _log(paths.dagger(col))) for col in collections(cfg)]
    if not any(log for _, log in logs):
        return False
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.8), dpi=130, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.09, hspace=0.12)
    fig.suptitle("Adaptive threshold, attempt by attempt, round after round", x=0.08, ha="left", fontsize=13, color=INK)
    note = ("After every attempt (successful or failed) p moves by (n / N)(q(M) - q(m)): m = mean VIG of the attempt's n "
            "policy steps (take-over trigger scores left out), N = policy steps so far, M = mean VIG of the round MoE's "
            "training frames. The line is the threshold after each update, starting at the round's start. Each round has "
            "its own MoE and reference, so the same percentile is a different VIG value from round to round.")
    fig.text(0.08, 0.915, textwrap.fill(note, 175), fontsize=8, color=MUTED, va="top")
    x0 = 0
    for i, (col, log) in enumerate(logs):
        if not log:
            continue
        color = ROUND_COLORS[i % len(ROUND_COLORS)]
        x = x0 + np.arange(0, len(log) + 1)
        a.plot(x, [log[0]["p_before"]] + [e["p_after"] for e in log], color=color, lw=1.6, drawstyle="steps-post",
               label=f"round {col.round} ({len(log)} attempts)")
        # the round's first update was made by an attempt that ran at the start threshold
        b.plot(x, [log[0]["used_threshold"]] + [e["threshold_after"] for e in log], color=color, lw=1.6, drawstyle="steps-post")
        if x0:
            for ax in (a, b):
                ax.axvline(x0, color=MUTED, lw=0.8, ls=":")
        x0 += len(log)
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    a.set_ylabel("threshold, percentile of\nthe round MoE's reference", color=MUTED, fontsize=9)
    b.set_ylabel("threshold (VIG)", color=MUTED, fontsize=9)
    b.set_xlabel("attempts so far (all rounds, in the order they finished)", color=MUTED, fontsize=9)
    a.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def round_scores(cfg: dict, col) -> dict | None:
    """A round's reference (its MoE's VIG on its training frames) and its collection's VIG, every attempt so far:
    the policy-step scores the threshold update uses (``adaptive.split_scores``) and the expert-step scores, with each
    attempt's threshold. None before the round's collection has an attempt."""
    import os

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py

    from vig_dagger.adaptive import percentile_of, split_scores

    paths = Paths.of(cfg)
    out, ref_path = paths.dagger(col), paths.demo_scores(col.moe)
    settings = _json(out / "settings.json")
    if settings is None or not ref_path.is_file():
        return None
    summaries = {}
    for path in sorted((out / "episodes").glob("worker_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                summaries[row["seed"]] = row
    take_count = int(settings["take_count"])
    exclude = bool((settings.get("adaptive") or {}).get("exclude_trigger", False))
    policy, expert, above, skipped = [], [], [0, 0], []
    for path in sorted((out / "attempts").glob("worker_*.h5")):
        try:
            with h5py.File(path, "r") as f:
                for name in f:
                    g = f[name]
                    s = summaries.get(int(g.attrs["seed"]))
                    if s is None:  # interrupted before its summary line
                        continue
                    p, e = split_scores(g["score"][:], g["controller"][:], g["take_steps"][:], take_count, exclude)
                    policy.append(p)
                    expert.append(e)
                    thr = float(s.get("threshold", settings["take_threshold"]))
                    above[0] += int((p > thr).sum())
                    above[1] += int((e > thr).sum())
        except OSError as exc:  # a worker writing it right now
            skipped.append(f"{path.name}: {exc}")
    if not policy:
        return None
    reference = np.sort(np.load(ref_path)["vig"].astype(float))
    pol, exp = np.concatenate(policy), np.concatenate(expert)
    return dict(reference=reference, policy=pol, expert=exp, attempts=len(policy),
                successes=sum(bool(s["success"]) for s in summaries.values()), skipped=skipped,
                policy_above=above[0] / max(len(pol), 1), expert_above=above[1] / max(len(exp), 1),
                reference_mean=float(reference.mean()), target_pct=percentile_of(reference, float(reference.mean())),
                policy_mean=float(pol.mean()) if len(pol) else None,
                policy_mean_pct=percentile_of(reference, float(pol.mean())) if len(pol) else None)


def _hist(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h = np.histogram(np.clip(x, bins[0], bins[-1] - 1e-12), bins)[0].astype(float)
    return h / max(h.sum(), 1.0)


def figure_distributions(cfg: dict, rows: list[dict], path: Path) -> dict[int, dict]:
    """One panel per round with collection data: VIG histograms of the MoE's training frames, the collection's policy
    steps, its policy + expert steps and its expert steps alone, with the overlap of the policy steps with the training frames. Returns per
    round the numbers drawn (empty: nothing to draw yet)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vig_dagger.adaptive import percentile_of

    by_round = {r["round"]: r for r in rows}
    data = {}
    for col in collections(cfg):
        d = round_scores(cfg, col)
        if d is not None:
            data[col.round] = d
    if not data:
        return {}
    n = len(data)
    width = max(9.5, 4.6 * n)
    left = 0.75 / width  # inches
    fig, axes = plt.subplots(1, n, figsize=(width, 5.8), dpi=130, squeeze=False)
    fig.subplots_adjust(left=left, right=0.985, top=0.70, bottom=0.11, wspace=0.18)
    fig.suptitle("VIG of each round MoE's training frames vs the VIG of its DAgger collection", x=left, ha="left",
                 fontsize=13, color=INK)
    note = ("Histograms scaled to area 1 (values past the right edge in the last bin). Grey: the round MoE's VIG on "
            "every 4th frame of all its training demos (the reference). Thick: the collection's policy steps, every "
            "attempt so far, successful and failed, as the threshold update counts them (scores while the policy drives, "
            "the 2 that trigger each take-over left out). Thin: the same plus the scores after expert steps (random-move "
            "steps are not scored). Dash-dot: the scores after expert steps alone. Overlap: the area the policy-step and "
            "training histograms share. Lines: the training "
            "mean (the update aims the policy steps' mean at it), the policy steps' mean, and the threshold now.")
    fig.text(left, 0.905, textwrap.fill(note, int(13.5 * width)), fontsize=7.8, color=MUTED, va="top")
    stats = {}
    for ax, (r, d) in zip(axes[0], sorted(data.items())):
        color = ROUND_COLORS[(r - 1) % len(ROUND_COLORS)]
        ref, pol, both = d["reference"], d["policy"], np.r_[d["policy"], d["expert"]]
        right = 1.1 * max(np.percentile(ref, 99.5), np.percentile(both, 99.5) if len(both) else 0.0)
        bins = np.linspace(0.0, right, 61)
        h_ref, h_pol, h_both = _hist(ref, bins), _hist(pol, bins), _hist(both, bins)
        overlap = float(np.minimum(h_pol, h_ref).sum()) if len(pol) else None
        overlap_both = float(np.minimum(h_both, h_ref).sum()) if len(both) else None
        width = bins[1] - bins[0]
        ax.stairs(h_ref / width, bins, color="#9ca3af", fill=True, alpha=0.45, label=f"training frames ({len(ref):,})")
        ax.stairs(h_pol / width, bins, color=color, lw=1.9, label=f"policy steps ({len(pol):,})")
        ax.stairs(h_both / width, bins, color=color, lw=0.9, alpha=0.75,
                  label=f"policy + expert steps ({len(both):,})")
        exp = np.asarray(d["expert"], float)
        exp_pct = percentile_of(np.sort(ref), float(exp.mean())) if len(exp) else None
        if len(exp):
            ax.stairs(_hist(exp, bins) / width, bins, color=color, lw=1.1, ls=(0, (3, 1.5, 1, 1.5)),
                      label=f"expert steps only ({len(exp):,}; mean p{exp_pct:.1f})")
        row = by_round.get(r, {})
        p_now = row.get("final_pct")
        lines = [(d["reference_mean"], "-", INK, f"training mean = p{d['target_pct']:.1f}")]
        if d["policy_mean"] is not None:
            lines.append((d["policy_mean"], (0, (4, 2)), color, f"policy-step mean = p{d['policy_mean_pct']:.1f}"))
        if p_now is not None:
            t_now = float(np.percentile(ref, p_now))
            lines.append((t_now, (0, (1, 1.5)), INK, f"threshold now = p{p_now:.1f}"))
        ymax = ax.get_ylim()[1]
        for (x, ls, c, label), y in zip(lines, (0.97, 0.88, 0.79)):
            x = min(x, right)
            ax.axvline(x, color=c, lw=1.2, ls=ls)
            on_left = x > 0.6 * right  # near the right edge: the label goes on the line's left
            ax.text(x + (-1 if on_left else 1) * right * 0.01, ymax * y, label, fontsize=7.5, color=INK, va="top",
                    ha="right" if on_left else "left", bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
        share = lambda x: "–" if x is None else f"{100 * x:.0f} %"  # noqa: E731
        ax.text(0.98, 0.62, f"overlap with the training frames:\npolicy steps {share(overlap)}, policy + expert "
                f"{share(overlap_both)}\nabove each attempt's threshold:\npolicy steps {share(d['policy_above'])}, "
                f"expert steps {share(d['expert_above'])}", transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
                color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
        start, demos = row.get("start_pct"), row.get("moe_training_demos")
        ax.set_title(f"Round {r}" + (f" · MoE on {demos} demos" if demos else "") +
                     f"\n{d['successes']} success{'es' if d['successes'] != 1 else ''} in {d['attempts']} attempts" +
                     (f" · threshold {_p(start)} → {_p(p_now)}" if start is not None else ""),
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xlim(0, right)
        ax.set_xlabel("VIG", color=MUTED, fontsize=9)
        ax.legend(fontsize=7.3, frameon=True, framealpha=0.9, edgecolor="none", loc=(0.38, 0.2))
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        stats[r] = dict(policy_steps=int(len(pol)), expert_steps=int(len(d["expert"])), expert_mean_pct=exp_pct,
                        policy_mean=d["policy_mean"],
                        policy_mean_pct=d["policy_mean_pct"], overlap=overlap, overlap_with_expert=overlap_both,
                        policy_above_threshold=d["policy_above"], expert_above_threshold=d["expert_above"])
        for s in d["skipped"]:
            print(f"[report] distributions, round {r}: skipped {s}", flush=True)
    axes[0][0].set_ylabel("density", color=MUTED, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return stats


def figure_rounds(rows: list[dict], path: Path, base0: dict | None = None) -> bool:
    """A: the collections per round; B: the policies on the evaluation scenes per round, with round 0 = the dense
    policy on the base demos alone (``base0``) where the dense and the planner-only lines both start."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not any(r.get("attempts") for r in rows):
        return False
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=130)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.27, wspace=0.25)
    fig.suptitle("Rounds: the collections, and the policies on the evaluation scenes", x=0.07, ha="left", fontsize=13, color=INK)
    x = np.array([r["round"] for r in rows])
    for key, label, color in (("success_rate", "collection success (per attempt)", "#2a78d6"),
                              ("expert_share", "expert share of the DAgger data", "#eb6834"),
                              ("move_share", "random-move share of the DAgger data", "#9ca3af")):
        y = [r.get(key) for r in rows]
        if any(v is not None for v in y):
            a.plot(x, [np.nan if v is None else 100 * v for v in y], marker="o", ms=6, lw=1.8, color=color, label=label)
    a.set_title("A · Collections", loc="left", fontsize=10, color=INK)
    a.set_ylabel("%", color=MUTED, fontsize=9)
    for key, label, color in (("dense", "dense policy retrained after the round (base + DAgger)", "#1baf7a"),
                              ("planner", "dense policy on as many planner demos", "#6b7280"),
                              ("moe", "the round's MoE", "#8e44ad")):
        ks = [(r["round"], r.get(f"{key}_successes"), r.get(f"{key}_episodes")) for r in rows if r.get(f"{key}_episodes")]
        if base0 and key in ("dense", "planner") and ks:
            ks = [(0, base0["successes"], base0["episodes"])] + ks
        if not ks:
            continue
        xs = np.array([k[0] for k in ks])
        rate = np.array([100 * k[1] / k[2] for k in ks])
        ci = np.array([[100 * v for v in wilson(k[1], k[2])] for k in ks])
        b.errorbar(xs, rate, yerr=[rate - ci[:, 0], ci[:, 1] - rate], marker="o", ms=6, lw=1.8, capsize=3, color=color, label=label)
    b.set_title("B · Success on the evaluation scenes (Wilson 95 %)", loc="left", fontsize=10, color=INK)
    b.set_ylabel("success (%)", color=MUTED, fontsize=9)
    for ax in (a, b):
        ax.set_xticks(np.r_[0, x] if ax is b and base0 else x)
        ax.set_xlabel("round" + (" (0: the base demos alone)" if ax is b and base0 else ""), color=MUTED, fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        if ax.lines:
            ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def _collection_entries(cfg: dict, rows: list[dict]) -> list[dict]:
    """Per analyzed round collection: ``summary.row`` (success, expert and random-move shares of the DAgger data, of all
    attempts and of the failed attempts) plus the takeovers per kept episode and the round's threshold and MoE."""
    from vig_dagger.summary import row as summary_row

    paths, by_round, out = Paths.of(cfg), {r["round"]: r for r in rows}, []
    for col in collections(cfg):
        res = _json(paths.collection_results(col).with_suffix(".json"))
        if not res:
            continue
        e = summary_row(res, col)
        takeovers = ((res["metrics"].get("data") or {}).get("per_episode") or {}).get("takeovers") or {}
        e.update(round=col.round, takeovers=takeovers.get("mean"), takeovers_std=takeovers.get("std"),
                 info=by_round.get(col.round, {}))
        out.append(e)
    return out


def figure_collections(cfg: dict, rows: list[dict], path: Path) -> bool:
    """One row per round's collection, as horizontal bars: success rate per attempt; expert share of the DAgger data, of
    all attempts and of the failed attempts (random-move steps stacked in orange); takeovers per kept episode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from vig_dagger.summary import EXPERT, MOVE, SUCCESS
    from vig_dagger.summary import GRID as SGRID
    from vig_dagger.summary import INK as SINK
    from vig_dagger.summary import INK2
    from vig_dagger.summary import MUTED as SMUTED

    entries = _collection_entries(cfg, rows)
    if not entries:
        return False
    d, n = cfg["dagger"], len(entries)
    tmax = max([(e["takeovers"] or 0.0) + (e["takeovers_std"] or 0.0) for e in entries] + [1.0])
    panels = [("success_rate", None, "Success rate per attempt", "Successes / all attempts", 1.45),
              ("data_expert", "data_move", "Expert share of the DAgger data", "Steps of the kept successful episodes", 1.3),
              ("all_expert", "all_move", "Expert share of all attempts", "Steps of every attempt, successful or failed",
               1.3),
              ("failed_expert", "failed_move", "Expert share of the failed attempts",
               "Steps of the failed attempts; grey: how many, steps each", 1.9),
              ("takeovers", None, "Takeovers per DAgger demo", "Mean over the kept episodes; whiskers: ±1 std",
               1.45 * tmax)]
    fig_h = 1.9 + 0.62 * n + 1.35
    fig, axes = plt.subplots(1, 5, figsize=(22, fig_h), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1.1, 1.1, 1.6, 0.9]))
    fig.subplots_adjust(left=0.115, right=0.985, top=1 - 1.75 / fig_h, bottom=1.3 / fig_h, wspace=0.13)
    ys = list(range(n - 1, -1, -1))
    for ax, (key, move_key, title, subtitle, xmax) in zip(axes, panels):
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        if key == "takeovers":
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
        else:
            ticks = [0, 0.25, 0.5, 0.75, 1.0]
            ax.set_xticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        ax.grid(axis="x", color=SGRID, zorder=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0, colors=SMUTED, labelsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
        for y, e in zip(ys, entries):
            value = e[key]
            if value is None:
                ax.text(0.01 * xmax, y, "–", va="center", fontsize=8.5, color=SMUTED)
                continue
            gray, end = "", value
            if key == "success_rate":
                ax.barh(y, value, 0.56, color=SUCCESS, zorder=2)
                bold, gray = _pct(value, 0).replace(" ", ""), f"{e['successes']}/{e['attempts']}"
            elif key == "takeovers":
                ax.barh(y, value, 0.56, color=EXPERT, zorder=2)
                std = e["takeovers_std"] or 0.0
                if std:
                    ax.errorbar(value, y, xerr=std, color=SINK, lw=1.0, capsize=4, zorder=3)
                bold, end = f"{value:.2f}" + (f" ± {std:.2f}" if std else ""), value + std
            else:
                move = e[move_key] or 0.0
                ax.barh(y, value, 0.56, color=EXPERT, zorder=2)
                if move:
                    ax.barh(y, move, 0.56, left=value, color=MOVE, zorder=2, edgecolor="white", linewidth=1.2)
                bold = f"{_pct(value, 0)} + {_pct(move, 0)}".replace(" %", "%") if move else _pct(value, 0).replace(" ", "")
                end = value + move
                if key == "failed_expert":
                    frames = (e["failed_expert_frames"] or 0.0) + (e["failed_move_frames"] or 0.0)
                    gray = f"{e['failed_episodes']} failed" + (f" · {frames:.0f} steps each" if e["failed_episodes"] else "")
            t = ax.annotate(bold, (end, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9.5,
                            fontweight="bold", color=SINK)
            if gray:
                fig.canvas.draw_idle()
                w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
                ax.annotate(gray, (end, y), xytext=(5 + w + 5, 0), textcoords="offset points", va="center", fontsize=8.3,
                            color=SMUTED)
    labels = []
    for e in entries:
        info = e["info"]
        demos = info.get("moe_training_demos")
        labels.append(f"Round {e['round']}" + (f" · MoE on {demos} demos" if demos else "")
                      + f"\nthreshold {_p(info.get('start_pct'))} → {_p(info.get('final_pct'))}")
    axes[0].set_yticks(ys, labels, fontsize=9.5)
    axes[1].legend(handles=[Patch(color=EXPERT, label="expert"),
                            Patch(color=MOVE, label="random move (removed before retraining)")],
                   loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, fontsize=8.8, frameon=False)
    title = (f"DAgger collections round by round: expert '{d['expert']}' takes over after {d['take_count']} VIG scores "
             f"above the adaptive threshold, the policy takes back control after {d['return_count']} below it")
    move = d["random_move"]
    note = (f"Each round's MoE: trained on the {cfg['demos']['num_demos']} base demos + the earlier rounds' DAgger data; "
            "threshold: where it started and where it ended (a percentile of that MoE's VIG on its training frames). "
            f"Random move: {move['radius'] * 100:g} cm / {move['rot_deg']:g}° at every take-over, before the expert "
            "(orange; cut out of the training data). Shares pooled over steps; the policy drove the rest. Kept "
            "successful episodes per round: " + ", ".join(str(x) for x in cfg["iterate"]["rounds"]) + ".")
    fig.text(0.01, 1 - 0.35 / fig_h, textwrap.fill(title, 190, break_on_hyphens=False), fontsize=12, fontweight="bold",
             color=SINK, va="top")
    fig.text(0.01, 0.12 / fig_h, textwrap.fill(note, 250, break_on_hyphens=False), fontsize=8.6, color=SMUTED,
             va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfa")
    plt.close(fig)
    return True


def figure_retraining(cfg: dict, rows: list[dict], path: Path, base0: dict | None = None) -> bool:
    """Success on the evaluation scenes per round, as horizontal bars with Wilson 95 % intervals: the dense policy
    retrained after the round (base demos + every round's DAgger data so far), the dense policy on as many planner
    demos (when its evaluation is in ``eval/planner/``), and the round's MoE; paired sign tests in the footnote."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vig_dagger.summary import GRID as SGRID
    from vig_dagger.summary import INK as SINK
    from vig_dagger.summary import INK2, SUCCESS
    from vig_dagger.summary import MUTED as SMUTED

    planner_color, moe_color = "#9ca3af", "#8e44ad"
    base, running, items = int(cfg["demos"]["num_demos"]), 0, []
    start = base_dense(cfg) if base0 is None else base0
    if start:
        items.append((f"Before DAgger: the {base} base demos", f"dense policy: the {base} base demos alone",
                      start["successes"], start["episodes"], "#52514e"))
    for r in rows:
        running += r["target"]
        group = f"Round {r['round']}: {base} base demos + {running} DAgger episodes"
        sub = [("dense policy: base + DAgger so far", r.get("dense_successes"), r.get("dense_episodes"), SUCCESS)]
        if r.get("planner_episodes"):
            sub.append((f"dense policy: {r['planner_demos']} planner demos only", r["planner_successes"],
                        r["planner_episodes"], planner_color))
        if r.get("moe_episodes"):
            sub.append((f"round {r['round']} MoE (collected this round's data)", r["moe_successes"], r["moe_episodes"],
                        moe_color))
        if any(s[2] for s in sub):
            items += [(group, *s) for s in sub]
    if not items:
        return False
    ypos, y, last, heads = [], 0.0, None, []
    for g, *_ in items:
        if g != last:
            y -= 1.25 if last is not None else 0.0
            heads.append((g, y))
            y -= 0.95
            last = g
        else:
            y -= 1.0
        ypos.append(y)
    fp = lambda x: "–" if x is None else f"{x:.2g}"  # noqa: E731
    tests = []
    for r in rows:
        k = r["round"]
        previous = f"round {k - 1} dense" if k > 1 else f"the {base} base demos' dense"
        for key, name, a, b in (("dense_vs_previous", f"round {k} dense vs {previous}", "only_this", "only_previous"),
                                ("dense_vs_planner", f"round {k} dense vs planner only", "only_dense", "only_planner"),
                                ("dense_vs_moe", f"round {k} dense vs round {k} MoE", "only_dense", "only_moe")):
            t = r.get(key)
            if t:
                tests.append(f"{name}: {t[a]} / {t[b]}, p = {fp(t['p'])}")
    ev = cfg["evaluation"]
    notes = [f"Dense: one full-resolution policy (no experts), {cfg['training']['epochs']} epochs, training seed "
             f"{cfg['retrain']['seeds'][0]}, random-move steps (and the 15 before each) cut from its DAgger data. Planner "
             "only: the same training on as many planner demos (the base demos + more collected like them; nested sets). "
             f"Same {ev['num_episodes']} scenes for every policy (from {ev['seed_start']}), CPU physics, "
             f"{ev['max_episode_steps']} steps, full resolution. Whiskers: Wilson 95 % interval."]
    if tests:
        notes.append("Paired sign tests (exact, uncorrected), scenes only the first / only the second solved: "
                     + "; ".join(tests) + ".")
    note_text = "\n".join(textwrap.fill(x, 165) for x in notes)
    n_note = note_text.count("\n") + 1
    fig_h = 1.5 + 0.42 * (max(1.0, -min(ypos) + 1.3)) + 0.2 * n_note + 0.5
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.subplots_adjust(left=0.36, right=0.97, top=1 - 1.25 / fig_h, bottom=(0.35 + 0.2 * n_note) / fig_h)
    ax.set_ylim(min(ypos) - 0.7, 0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ["0%", "20%", "40%", "60%", "80%"])
    ax.grid(axis="x", color=SGRID, zorder=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, colors=SMUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    for g, gy in heads:
        ax.text(-0.555, gy, g, transform=ax.get_yaxis_transform(), fontsize=10.5, fontweight="bold", color=SINK,
                va="center")
    for (g, label, k, total, color), yy in zip(items, ypos):
        if not total:
            ax.text(0.01, yy, "not evaluated yet", va="center", fontsize=8.5, color=SMUTED)
            continue
        v = k / total
        lo, hi = wilson(k, total)
        ax.barh(yy, v, 0.56, color=color, zorder=2)
        ax.plot([lo, hi], [yy, yy], color=INK2, lw=1.1, zorder=4)
        for q in (lo, hi):
            ax.plot([q, q], [yy - 0.13, yy + 0.13], color=INK2, lw=1.1, zorder=4)
        t = ax.annotate(f"{100 * v:.1f}%", (hi, yy), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9.5, fontweight="bold", color=SINK)
        fig.canvas.draw_idle()
        w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
        ax.annotate(f"{k}/{total}", (hi, yy), xytext=(6 + w + 6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=SMUTED)
    ax.set_yticks(ypos, [it[1] for it in items], fontsize=9.5)
    fig.text(0.015, 1 - 0.3 / fig_h, "Success on the evaluation scenes, round by round", fontsize=13, fontweight="bold",
             color=SINK, va="top")
    fig.text(0.015, 1 - 0.72 / fig_h, "The dense policy retrained after each round, the dense policy on as many planner "
             "demos, and the round's MoE.", fontsize=9.5, color=INK2, va="top")
    fig.text(0.015, 0.2 / fig_h, note_text, fontsize=8.3, color=SMUTED, va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfa")
    plt.close(fig)
    return True


def _write_planner_only(cfg: dict, rows: list[dict], results: Path) -> dict:
    """``iterate.planner_baselines: only``: the planner-only dense policies alone (their DAgger counterparts run
    elsewhere; put both evaluations in one eval/ folder and run the report there to compare them)."""
    keep = ("round", "planner_demos", "planner_successes", "planner_episodes", "planner_success_rate")
    base0 = base_dense(cfg)
    summary = dict(planner_only=[{k: r.get(k) for k in keep} for r in rows], base_dense=base0)
    (results / "iterate_summary.json").write_text(json.dumps(summary, indent=1))
    ev = cfg["evaluation"]
    lines = [f"# {cfg['name']}: results", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{paths_root(cfg)}`. Regenerate with `vig-dagger report`.", "",
             "Planner-only dense policies (`iterate.planner_baselines: only`): per round of the DAgger experiment, a dense "
             "full-resolution policy trained like that round's dense policy on as many planner demos (the base demos + "
             "more collected like them; the sets are nested), evaluated on the same scenes "
             f"({ev['num_episodes']} from {ev['seed_start']}, full resolution, CPU physics).", "",
             "| round | planner demos | dense: planner only |", "|---|---|---|"]
    lines.append(f"| 0 | {cfg['demos']['num_demos']} (the base demos alone) | "
                 + (f"{base0['successes']}/{base0['episodes']} = {_pct(base0['success_rate'])} |" if base0 else "– |"))
    for r in rows:
        res = (f"{r['planner_successes']}/{r['planner_episodes']} = {_pct(r['planner_success_rate'])}"
               if r.get("planner_episodes") else "–")
        lines.append(f"| {r['round']} | {r['planner_demos']} | {res} |")
    (results / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def paths_root(cfg: dict) -> Path:
    return Paths.of(cfg).root


def write(cfg: dict) -> dict:
    paths = Paths.of(cfg)
    results, figures = paths.results(), paths.results() / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = rounds(cfg)
    it, d = cfg["iterate"], cfg["dagger"]
    summary = dict(rounds=rows, settings=dict(rounds=it["rounds"], start_percentile=it["start_percentile"],
                                              exclude_trigger=it["exclude_trigger"], expert=d["expert"],
                                              variant=d["variants"][0], base_demos=cfg["demos"]["num_demos"],
                                              base_motion=cfg["demos"]["motion"]))
    if cfg["iterate"]["planner_baselines"] == "only":
        return _write_planner_only(cfg, rows, results)
    distributions = figure_distributions(cfg, rows, figures / "distributions.png")
    for row in rows:
        row["distribution"] = distributions.get(row["round"])
    base0 = base_dense(cfg)
    summary["base_dense"] = base0
    (results / "iterate_summary.json").write_text(json.dumps(summary, indent=1))
    made = [name for name, ok in (("collections.png", figure_collections(cfg, rows, figures / "collections.png")),
                                  ("retraining.png", figure_retraining(cfg, rows, figures / "retraining.png", base0)),
                                  ("thresholds.png", figure_thresholds(cfg, figures / "thresholds.png")),
                                  ("rounds.png", figure_rounds(rows, figures / "rounds.png", base0)),
                                  ("distributions.png", bool(distributions))) if ok]
    from vig_dagger import move_figures

    made += move_figures.round_figures(cfg, figures)  # moves/spheres_{held,progress}_rounds.png
    lines = [f"# {cfg['name']}: results", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{paths.root}`. Regenerate with `vig-dagger report`.", "",
             f"{len(rows)} rounds of DAgger ({', '.join(str(n) for n in it['rounds'])} successful episodes), each driven by "
             f"a MoE trained on the {cfg['demos']['num_demos']} base demos (planner, motion '{cfg['demos']['motion']}') plus "
             f"the earlier rounds' DAgger data, expert '{d['expert']}', variant {d['variants'][0].replace('_', ' ')}. One "
             f"threshold for take-over ({d['take_count']} scores above) and hand-back ({d['return_count']} below), adapted "
             f"after every attempt, starting at p{it['start_percentile']:g} in round 1 and where the previous round ended "
             "afterwards. After each round a dense full-resolution policy is retrained on the base demos plus every round's "
             "DAgger data so far (random-move steps cut).", "",
             "## Rounds", "",
             "| round | MoE training demos | reference frames | M (mean VIG) | target q(M) | threshold start → end | "
             "second half: median (10–90 %) | attempts | successes | expert + move share of the data |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        half = (f"{_p(r.get('second_half_median_pct'))} ({_p(r.get('second_half_p10'))}–{_p(r.get('second_half_p90'))})"
                if r.get("second_half_median_pct") is not None else "–")
        ref = f"{r['reference_mean']:.4g}" if r.get("reference_mean") is not None else "–"
        end = _p(r["final_pct"]) + (f" = {r['final_threshold']:.4g}" if r.get("final_threshold") is not None else "")
        lines.append(f"| {r['round']} | {r['moe_training_demos'] or '–'} | {r['reference_frames'] or '–'} | {ref} | "
                     f"{_p(r['target_pct'])} | {_p(r['start_pct'])} → {end} | {half} | {r['attempts'] or '–'} | "
                     f"{r['successes'] if r['successes'] is not None else '–'} (target {r['target']}) | "
                     f"{_pct(r['expert_share'])} + {_pct(r['move_share'])} |")
    lines += ["", "## Policies on the evaluation scenes", "",
              f"{cfg['evaluation']['num_episodes']} scenes from {cfg['evaluation']['seed_start']}, full resolution, CPU "
              "physics. Paired tests: scenes only one of the two solves, exact sign test.", "",
              "| round | dense: base + DAgger so far | dense success | vs previous round's dense (this / previous) | "
              "round MoE success | dense vs MoE (dense / MoE) |", "|---|---|---|---|---|---|"]
    if base0:
        lines.append(f"| 0 | {base0['demos']} (base demos alone) | {base0['successes']}/{base0['episodes']} = "
                     f"{_pct(base0['success_rate'])} | – | – | – |")
    running = 0
    for r in rows:
        running += r["target"]
        dense = (f"{r['dense_successes']}/{r['dense_episodes']} = {_pct(r['dense_success_rate'])}"
                 if r.get("dense_episodes") else "–")
        moe = f"{r['moe_successes']}/{r['moe_episodes']} = {_pct(r['moe_success_rate'])}" if r.get("moe_episodes") else "–"
        lines.append(f"| {r['round']} | {cfg['demos']['num_demos']} + {running} | {dense} | "
                     f"{_test(r.get('dense_vs_previous'), 'only_this', 'only_previous')} | {moe} | "
                     f"{_test(r.get('dense_vs_moe'), 'only_dense', 'only_moe')} |")
    if cfg["iterate"]["planner_baselines"] or any(r.get("planner_episodes") for r in rows):
        lines += ["", "## DAgger against planner demos only", "",
                  "After each round, a dense policy trained the same way on as many planner demos as the round's dense "
                  "policy trains on (the base demos + more planner demos collected like them; the sets are nested), on "
                  "the same scenes. Paired test: scenes only one of the two solves, exact sign test.", "",
                  "| round | demos | dense: base + DAgger | dense: planner only | DAgger / planner (scenes only one solves), p |",
                  "|---|---|---|---|---|"]
        for r in rows:
            dense = (f"{r['dense_successes']}/{r['dense_episodes']} = {_pct(r['dense_success_rate'])}"
                     if r.get("dense_episodes") else "–")
            planner = (f"{r['planner_successes']}/{r['planner_episodes']} = {_pct(r['planner_success_rate'])}"
                       if r.get("planner_episodes") else "–")
            lines.append(f"| {r['round']} | {r['planner_demos']} | {dense} | {planner} | "
                         f"{_test(r.get('dense_vs_planner'), 'only_dense', 'only_planner')} |")
    if made:
        lines += ["", "## Figures", ""] + [f"![{m}](figures/{m})" for m in made]
    (results / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary
