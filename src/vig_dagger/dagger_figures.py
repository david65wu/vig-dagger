"""Figures of a round's DAgger collections per task stage, like the stage analysis of evaluation rollouts.

Built by ``report.py`` from what the analysis writes to ``results/``: per scorer, ``dagger_<round>_<scorer>_steps.npz``
(every attempt's stage, driver, score and flags per step) and ``_takeovers.csv`` (every takeover), plus the round's
demo-frame scores (``policies/drivers/<round>/demo_scores.npz``, where the thresholds come from). Attempts are grouped
into all, successful and failed ones; a step belongs to the stage of the observation it was taken from.

  dagger_<round>_stage_outcomes.png   per stage: attempts that reached it and failed there, the expert's share of its
                                      steps, steps spent there, and the failed attempts by reason
  dagger_<round>_stage_scores.png     the score per stage on policy- and expert-driven steps (p5, p25, median, p75,
                                      p95, mean), with the demo frames and both thresholds
  dagger_<round>_stage_take.png       taking over, per stage: policy-driven steps above the take threshold, attempts
                                      with a run above it and with a takeover, takeovers per 100 policy-driven steps,
                                      and where in the stage takeovers start
  dagger_<round>_stage_return.png     handing back, per stage: expert-driven steps below the return threshold, and each
                                      takeover's longest run below it against the run a hand-back needs
  dagger_<round>_flags.png            the failure flags: attempts with each flag, the expert's share of flagged steps,
                                      and policy-driven flagged steps above the take threshold
  dagger_<round>_takeover_stages.png  the stage each takeover started in against where it ended
  dagger_<round>_takeover_timing.png  takeover and hand-back steps, the expert's share per step, the first takeover's
                                      stage
  dagger_<round>_episodes.png         sampled attempts step by step: stage (color) and expert driving (black line)

Over every collection of every round: ``dagger_takeovers_by_stage.png``, box plots per stage of the takeovers per
attempt and of the takeover lengths.

``dagger_<round>_<scorer>_by_stage.csv`` holds the per-stage numbers behind the stage figures, for every group.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vig_dagger import stages
from vig_dagger.config import Paths

POLICY, EXPERT = 0, 1  # the controller codes of recording.py
STAGE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")  # up to 7 stages
PHASE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")  # longer tasks: one hue per phase (stages.PHASES)
INK, INK_MUTED, PALE = "#0b0b0b", "#52514e", "#d9d6ce"
GROUPS = (("all", "all attempts", "#2b2a27"), ("success", "successful", "#858279"), ("failed", "failed", "#c23434"))
DRIVERS = {"policy": "#3b6ea5", "expert": "#e8a33d"}
REASON_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
SHADES = ("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _mix(color: str, other: str, t: float) -> str:
    a, b = (np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in (color, other))
    return "#" + "".join(f"{int(round(v)):02x}" for v in a + (b - a) * t)


def stage_colors() -> list[str]:
    """One color per stage: the categorical slots in order for up to 7 stages; for a longer task, one hue per phase
    (``stages.PHASES``: (name, first stage) pairs), from light at the phase's first stage to full at its last."""
    phases = getattr(stages, "PHASES", None)
    if not phases:
        return list(STAGE_COLORS[:len(stages.STAGES)])
    bounds = [first for _, first in phases] + [len(stages.STAGES)]
    return [_mix(hue, "#ffffff", 0.6 * (b - 1 - k) / max(1, b - 1 - a))
            for hue, a, b in zip(PHASE_COLORS, bounds, bounds[1:]) for k in range(a, b)]


COL_W = max(7.6, 0.6 * len(stages.STAGES))  # width of one scorer's column of panels


def _stage_ticks(ax, positions, names) -> None:
    many = len(names) > 8
    ax.set_xticks(positions, names, fontsize=7.5 if many else 8, rotation=30 if many else 0, ha="right" if many else "center")


def _start_stages(tos_lists) -> list[str]:
    """The stages a takeover can start in: all but the last (the task's success), unless one did start there."""
    last = len(stages.STAGES) - 1
    return list(stages.STAGES[:last + any(t["stage"] in (last, stages.STAGES[last]) for tos in tos_lists for t in tos)])


def _runs(mask) -> list[tuple[int, int]]:
    """(start, length) of every run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def _group(rows: list[dict], group: str) -> list[dict]:
    return rows if group == "all" else [a for a in rows if a["success"] == (group == "success")]


def load_steps(path: Path) -> list[dict]:
    """Every attempt's per-step record (``dagger_<round>_<scorer>_steps.npz``), in seed order."""
    if not path.exists():
        return []
    z = np.load(path)
    o = z["offsets"]
    rows = [dict(seed=int(z["seed"][i]), success=bool(z["success"][i]), reason=str(z["reason"][i]),
                 furthest=int(z["furthest"][i]), **{k: z[k][o[i]:o[i + 1]] for k in ("stage", "controller", "score", "flags")})
            for i in range(len(z["seed"]))]
    return sorted(rows, key=lambda a: a["seed"])


def takeovers(rows: list[dict], settings: dict) -> list[dict]:
    """Every takeover: its attempt, start, length, stage at the start, where in that stage it started (0 = the
    attempt's first step in the stage, 1 = its last), whether it handed back, and its longest run of scores below the
    return threshold. Those are the scores the gate checks for a hand-back: of the observation after each expert step,
    the last of them the one the policy resumes from; unscored steps (e.g. an atomic grasp) neither count nor break a run."""
    out = []
    for a in rows:
        for start, length in _runs(a["controller"] == EXPERT):
            k = int(a["stage"][start])
            idx = np.flatnonzero(a["stage"] == k)
            checked = a["score"][start + 1:start + length + 1]
            below = _runs(checked[np.isfinite(checked)] < settings["return_threshold"])
            out.append(dict(seed=a["seed"], success=a["success"], start=start, length=length, stage=k,
                            position=(start - idx[0]) / (idx[-1] - idx[0] + 1),
                            returned=start + length < len(a["controller"]),
                            longest_below=max([n for _, n in below], default=0)))
    return out


def _quantiles(values) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, p5=None, p25=None, median=None, p75=None, p95=None, mean=None)
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=int(len(x)), p5=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p95=float(q[4]),
                mean=float(x.mean()))


def _share(num, den):
    return num / den if den else None


def stage_stats(rows: list[dict], settings: dict) -> list[dict]:
    """One row per group of attempts and stage: the numbers behind the stage figures."""
    take, ret = settings["take_threshold"], settings["return_threshold"]
    tos = takeovers(rows, settings)
    out = []
    for group, _, _ in GROUPS:
        att = _group(rows, group)
        seeds = {a["seed"] for a in att}
        for k, name in enumerate(stages.STAGES):
            here = [a["stage"] == k for a in att]
            pol = [m & (a["controller"] == POLICY) for m, a in zip(here, att)]
            exp = [m & (a["controller"] == EXPERT) for m, a in zip(here, att)]
            steps = np.array([int(m.sum()) for m in here])
            p_steps, e_steps = sum(int(m.sum()) for m in pol), sum(int(m.sum()) for m in exp)
            above = sum(int((a["score"][m] > take).sum()) for a, m in zip(att, pol))
            below = sum(int((a["score"][m] < ret).sum()) for a, m in zip(att, exp))
            with_policy = [i for i, m in enumerate(pol) if m.any()]
            run = sum(bool(_runs(pol[i] & (att[i]["score"] > take))) for i in with_policy)
            started = [t for t in tos if t["seed"] in seeds and t["stage"] == k]
            taken = len({t["seed"] for t in started} & {att[i]["seed"] for i in with_policy})
            visited = steps[steps > 0]
            spent = np.percentile(visited, [25, 50, 75]) if len(visited) else [None] * 3
            longest = _quantiles([t["longest_below"] for t in started])
            row = dict(group=group, stage=name, attempts=len(att), reached=sum(a["furthest"] >= k for a in att),
                       failed_here=sum(not a["success"] and a["furthest"] == k for a in att), visited=int(len(visited)),
                       steps=int(steps.sum()), steps_p25=spent[0], steps_median=spent[1], steps_p75=spent[2],
                       policy_steps=p_steps, expert_steps=e_steps, expert_share=_share(e_steps, p_steps + e_steps),
                       policy_above_take=_share(above, p_steps), expert_below_return=_share(below, e_steps),
                       attempts_with_policy_steps=len(with_policy), with_run_above_take=_share(run, len(with_policy)),
                       with_takeover=_share(taken, len(with_policy)), takeovers=len(started),
                       takeovers_per_100_policy_steps=_share(100 * len(started), p_steps),
                       returned=sum(t["returned"] for t in started), **{f"longest_below_{q}": v for q, v in longest.items()})
            for who, masks in (("policy", pol), ("expert", exp)):
                q = _quantiles(np.concatenate([a["score"][m] for a, m in zip(att, masks)]) if att else [])
                row.update({f"{who}_score_{key}": v for key, v in q.items()})
            row["fail_rate"] = _share(row["failed_here"], row["reached"])
            out.append(row)
    return out


def flag_stats(rows: list[dict], settings: dict) -> list[dict]:
    """One row per group of attempts and failure flag."""
    out = []
    for group, _, _ in GROUPS:
        att = _group(rows, group)
        for i, name in enumerate(stages.FLAGS):
            on = [((a["flags"] >> i) & 1).astype(bool) for a in att]
            pol = [m & (a["controller"] == POLICY) for m, a in zip(on, att)]
            steps, e_steps, p_steps = (sum(int(m.sum()) for m in on), sum(int((m & (a["controller"] == EXPERT)).sum()) for m, a in zip(on, att)),
                                       sum(int(m.sum()) for m in pol))
            above = sum(int((a["score"][m] > settings["take_threshold"]).sum()) for a, m in zip(att, pol))
            out.append(dict(group=group, flag=name, attempts=len(att), with_flag=_share(sum(bool(m.any()) for m in on), len(att)),
                            steps=steps, expert_share=_share(e_steps, steps), policy_above_take=_share(above, p_steps)))
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pick(table: list[dict], group: str, key: str) -> list:
    return [r[key] for r in table if r["group"] == group]


def _grouped_bars(ax, labels, series, ylim=None) -> None:
    """Side-by-side bars per label for each (name, color, values); missing values leave a gap."""
    x, w = np.arange(len(labels)), 0.8 / len(series)
    for i, (name, color, values) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, [np.nan if v is None else v for v in values], w * 0.9,
               color=color, label=name)
    _stage_ticks(ax, x, labels)
    ax.set_ylim(*(ylim or (0, None)))


def _finish(plt, fig, figures: Path, name: str, title: str) -> str:
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / name, dpi=90)
    plt.close(fig)
    return name


def _by_group(table: list[dict], key: str) -> list[tuple]:
    return [(label, color, _pick(table, g, key)) for g, label, color in GROUPS]


# ---------------------------------------------------------------- stage figures


def stage_outcomes_figure(plt, figures: Path, r: str, data: dict) -> str:
    """Per scorer (columns): attempts that reached each stage and failed there, the expert's share of the stage's
    steps, steps spent in the stage, and the failed attempts by furthest stage and reason."""
    names = stages.STAGES
    fig, axes = plt.subplots(4, len(data), figsize=(COL_W * len(data), 11.6), squeeze=False)
    reasons = sorted({a["reason"] for d in data.values() for a in d["rows"] if not a["success"]})
    for col, (s, d) in enumerate(data.items()):
        t = d["table"]
        reached, failed = _pick(t, "all", "reached"), _pick(t, "all", "failed_here")
        ax = axes[0, col]
        x = np.arange(len(names))
        ax.bar(x, reached, 0.7, color=PALE, label="reached the stage")
        ax.bar(x, failed, 0.7, color=GROUPS[2][2], label="failed there (furthest stage)")
        for i, (n, f) in enumerate(zip(reached, failed)):
            if f:
                ax.text(i, n, f"{f}/{n}\n{100 * f / n:.0f}%", ha="center", va="bottom", fontsize=7, color=INK)
        _stage_ticks(ax, x, names); ax.set_ylabel("attempts", fontsize=8); ax.margins(y=0.18)
        ax.set_title(f"{s}: attempts that reached each stage, and failed there ({len(d['rows'])} attempts)", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[1, col]
        _grouped_bars(ax, names, _by_group(t, "expert_share"), ylim=(0, 1))
        ax.set_title(f"{s}: share of the stage's steps the expert drove", fontsize=9); ax.set_ylabel("expert share", fontsize=8)
        ax.legend(fontsize=7)

        ax = axes[2, col]
        w = 0.8 / len(GROUPS)
        for i, (g, label, color) in enumerate(GROUPS):
            med = np.array([np.nan if v is None else v for v in _pick(t, g, "steps_median")], float)
            lo = med - np.array([np.nan if v is None else v for v in _pick(t, g, "steps_p25")], float)
            hi = np.array([np.nan if v is None else v for v in _pick(t, g, "steps_p75")], float) - med
            ax.bar(x + (i - 1) * w, med, w * 0.9, color=color, label=label, yerr=[lo, hi], capsize=2,
                   error_kw=dict(lw=1, ecolor=INK_MUTED))
        _stage_ticks(ax, x, names); ax.set_ylabel("steps", fontsize=8); ax.legend(fontsize=7)
        ax.set_title(f"{s}: steps spent in the stage per attempt that got there (median, p25–p75)", fontsize=9)

        ax = axes[3, col]
        bottom = np.zeros(len(names))
        for reason, color in zip(reasons, REASON_COLORS * 2):
            counts = np.array([sum(1 for a in d["rows"] if not a["success"] and a["reason"] == reason and a["furthest"] == k)
                               for k in range(len(names))])
            ax.bar(x, counts, 0.7, bottom=bottom, color=color, label=reason)
            bottom += counts
        _stage_ticks(ax, x, names); ax.set_ylabel("failed attempts", fontsize=8); ax.set_ylim(0, max(1, bottom.max() * 1.1))
        ax.set_title(f"{s}: failed attempts by furthest stage and reason", fontsize=9)
        if bottom.any():
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no failed attempts", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=INK_MUTED)
    return _finish(plt, fig, figures, f"dagger_{r}_stage_outcomes.png", f"round {r}: how far DAgger attempts get, stage by stage")


def stage_scores_figure(plt, figures: Path, r: str, data: dict, demo: dict) -> str:
    """Per scorer: box plots of the score in each stage, policy- and expert-driven steps apart, under the demo frames'
    distribution; log scale; both thresholds as vertical lines."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(data), figsize=(COL_W * len(data), max(7.4, 1.2 + 0.8 * (len(stages.STAGES) + 1))), squeeze=False)
    for ax, (s, d) in zip(axes[0], data.items()):
        st, t = d["settings"], d["table"]
        rows = [row for row in t if row["group"] == "all"]
        floor = 1e-4  # the log axis needs positive values
        stats, positions, colors, labels = [], [], [], []

        def box(q: dict, y: float, color: str) -> None:
            if q["n"]:
                stats.append(dict(med=max(q["median"], floor), q1=max(q["p25"], floor), q3=max(q["p75"], floor),
                                  whislo=max(q["p5"], floor), whishi=max(q["p95"], floor), mean=max(q["mean"], floor), fliers=[]))
                positions.append(y)
                colors.append(color)

        if s in demo:
            box(_quantiles(demo[s]), 0, "none")
            labels.append(f"demo frames ({len(demo[s])})")
        else:
            labels.append("demo frames (–)")
        for k, row in enumerate(rows, start=1):
            for who, dy in (("policy", -0.2), ("expert", 0.2)):
                box({key: row[f"{who}_score_{key}"] for key in ("n", "p5", "p25", "median", "p75", "p95", "mean")}, k + dy, DRIVERS[who])
            labels.append(f"{row['stage']}\npolicy {row['policy_score_n']:,} · expert {row['expert_score_n']:,}")
        if stats:
            parts = ax.bxp(stats, positions=positions, orientation="horizontal", widths=0.34, patch_artist=True, showmeans=True,
                           showcaps=False, meanprops=dict(marker="s", markerfacecolor="white", markeredgecolor=INK, markersize=5),
                           medianprops=dict(color="white", lw=1.6), whiskerprops=dict(color=INK_MUTED, lw=1.2))
            for patch, color in zip(parts["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor(INK if color == "none" else color)
            for med, color in zip(parts["medians"], colors):
                if color == "none":
                    med.set_color(INK)
        ax.set_xscale("log")
        ax.axvline(st["take_threshold"], color=INK, lw=1.3)
        ax.axvline(st["return_threshold"], color=INK, lw=1.3, ls="--")
        ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlabel(f"{s} score (log scale)", fontsize=8)
        ax.set_title(f"{s}: score per stage", fontsize=9)
        handles = [Patch(color=DRIVERS["policy"], label="policy-driven steps"), Patch(color=DRIVERS["expert"], label="expert-driven steps"),
                   Patch(facecolor="none", edgecolor=INK, label="demo frames"),
                   Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=INK, label="mean"),
                   Line2D([], [], color=INK, lw=1.3, label=f"take {st['take_spec']} ({st['take_threshold']:.4g})"),
                   Line2D([], [], color=INK, lw=1.3, ls="--", label=f"return {st['return_spec']} ({st['return_threshold']:.4g})")]
        ax.legend(handles=handles, fontsize=7, loc="lower right")
    return _finish(plt, fig, figures, f"dagger_{r}_stage_scores.png",
                   f"round {r}: the score in each stage (box p25–p75 split at the median, whiskers p5–p95, all attempts)")


def stage_take_figure(plt, figures: Path, r: str, data: dict, demo: dict) -> str:
    """Taking over, per stage and scorer: policy-driven steps above the take threshold, attempts with a run above it
    and with a takeover, takeovers per 100 policy-driven steps, and where in the stage takeovers start."""
    from matplotlib.colors import LinearSegmentedColormap

    names = stages.STAGES
    cmap = LinearSegmentedColormap.from_list("share", SHADES)
    fig, axes = plt.subplots(4, len(data), figsize=(COL_W * len(data), 9.6 + 0.25 * len(names)), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, max(1.0, len(names) / 7)]))
    for col, (s, d) in enumerate(data.items()):
        st, t = d["settings"], d["table"]
        ax = axes[0, col]
        _grouped_bars(ax, names, _by_group(t, "policy_above_take"))
        if s in demo:
            ax.axhline(float(np.mean(demo[s] > st["take_threshold"])), color=INK, lw=1, ls=":", label="demo frames above it")
        ax.set_ylabel("share of steps", fontsize=8); ax.legend(fontsize=7)
        ax.set_title(f"{s}: policy-driven steps above the take threshold ({st['take_spec']})", fontsize=9)

        ax = axes[1, col]
        _grouped_bars(ax, names, [("a run of steps above the threshold", PALE, _pick(t, "all", "with_run_above_take")),
                                  (f"a takeover ({st['take_count']} steps in a row)", GROUPS[0][2], _pick(t, "all", "with_takeover"))],
                      ylim=(0, 1))
        ax.set_ylabel("share of attempts", fontsize=8); ax.legend(fontsize=7)
        ax.set_title(f"{s}: attempts with policy-driven steps in the stage that had …", fontsize=9)

        ax = axes[2, col]
        _grouped_bars(ax, names, _by_group(t, "takeovers_per_100_policy_steps"))
        ax.set_ylabel("takeovers", fontsize=8); ax.legend(fontsize=7)
        ax.set_title(f"{s}: takeovers per 100 policy-driven steps in the stage", fontsize=9)

        ax = axes[3, col]
        heat = np.zeros((len(names), 10))
        for tk in d["takeovers"]:
            heat[tk["stage"], min(9, int(10 * tk["position"]))] += 1
        share = heat / np.maximum(heat.sum(1, keepdims=True), 1)
        ax.imshow(np.ma.masked_where(heat == 0, share), cmap=cmap, vmin=0, vmax=max(0.3, float(share.max())), aspect="auto")
        for i, j in zip(*np.nonzero(heat)):
            ax.text(j, i, f"{int(heat[i, j])}", ha="center", va="center", fontsize=7, color="white" if share[i, j] > 0.2 else INK)
        ax.set_xticks(range(10), [f"{10 * i}–{10 * i + 10}%" for i in range(10)], fontsize=7)
        ax.set_yticks(range(len(names)), [f"{n} ({int(c)})" for n, c in zip(names, heat.sum(1))], fontsize=8)
        ax.set_xlabel("time from the attempt's first step in the stage to its last", fontsize=8)
        ax.set_title(f"{s}: where in the stage takeovers start (count; shade: share of the row)", fontsize=9)
    return _finish(plt, fig, figures, f"dagger_{r}_stage_take.png", f"round {r}: the expert taking over, stage by stage")


def stage_return_figure(plt, figures: Path, r: str, data: dict, demo: dict) -> str:
    """Handing back, per stage and scorer: expert-driven steps below the return threshold, and each takeover's longest
    run below it (by the stage the takeover started in) against the run a hand-back needs."""
    names = stages.STAGES
    fig, axes = plt.subplots(2, len(data), figsize=(COL_W * len(data), 7.2), squeeze=False)
    for col, (s, d) in enumerate(data.items()):
        st, t = d["settings"], d["table"]
        ax = axes[0, col]
        _grouped_bars(ax, names, _by_group(t, "expert_below_return"), ylim=(0, 1))
        if s in demo:
            ax.axhline(float(np.mean(demo[s] < st["return_threshold"])), color=INK, lw=1, ls=":", label="demo frames below it")
        ax.set_ylabel("share of steps", fontsize=8); ax.legend(fontsize=7)
        ax.set_title(f"{s}: expert-driven steps below the return threshold ({st['return_spec']})", fontsize=9)

        ax = axes[1, col]
        rows = [row for row in t if row["group"] == "all"]
        stats, positions = [], []
        need = int(st["return_count"])
        for k, row in enumerate(rows):
            if row["longest_below_n"]:
                stats.append(dict(med=row["longest_below_median"], q1=row["longest_below_p25"], q3=row["longest_below_p75"],
                                  whislo=row["longest_below_p5"], whishi=row["longest_below_p95"], fliers=[]))
                positions.append(k)
                ax.text(k, max(row["longest_below_p95"], need) + 1, f"{row['returned']}/{row['takeovers']} returned",
                        ha="center", va="bottom", fontsize=7, color=INK)
        if stats:
            parts = ax.bxp(stats, positions=positions, widths=0.5, patch_artist=True, showcaps=False,
                           medianprops=dict(color="white", lw=1.6), whiskerprops=dict(color=INK_MUTED, lw=1.2))
            for patch in parts["boxes"]:
                patch.set_facecolor(DRIVERS["expert"]); patch.set_edgecolor(DRIVERS["expert"])
        ax.axhline(need, color=INK, lw=1.3, ls="--", label=f"a hand-back needs {need} in a row")
        _stage_ticks(ax, range(len(names)), [f"{n} ({row['takeovers']})" for n, row in zip(names, rows)])
        ax.set_xlim(-0.6, len(names) - 0.4); ax.set_ylim(0, need * 1.5)
        ax.set_ylabel("steps in a row", fontsize=8); ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{s}: per takeover, the longest run below the return threshold, by start stage (takeovers)", fontsize=9)
    return _finish(plt, fig, figures, f"dagger_{r}_stage_return.png",
                   f"round {r}: the expert handing back, stage by stage (boxes: p25–p75 split at the median, whiskers p5–p95)")


def flags_figure(plt, figures: Path, r: str, data: dict) -> str:
    """Per failure flag and scorer: attempts with the flag, the expert's share of the flagged steps, and the share of
    policy-driven flagged steps above the take threshold (dotted: over every policy-driven step)."""
    names = [k.replace("_", "-") for k in stages.FLAGS]
    fig, axes = plt.subplots(3, len(data), figsize=(COL_W * len(data), 8.6), squeeze=False)
    for col, (s, d) in enumerate(data.items()):
        f, st = d["flags"], d["settings"]
        _grouped_bars(axes[0, col], names, _by_group(f, "with_flag"), ylim=(0, 1))
        axes[0, col].set_title(f"{s}: attempts with the flag at some step", fontsize=9)
        _grouped_bars(axes[1, col], names, _by_group(f, "expert_share"), ylim=(0, 1))
        axes[1, col].set_title(f"{s}: share of the flagged steps the expert drove", fontsize=9)
        _grouped_bars(axes[2, col], names, _by_group(f, "policy_above_take"))
        policy = [a["score"][a["controller"] == POLICY] for a in d["rows"]]
        policy = np.concatenate(policy) if policy else np.zeros(0)
        if len(policy):
            axes[2, col].axhline(float(np.mean(policy > st["take_threshold"])), color=INK, lw=1, ls=":",
                                 label="every policy-driven step")
        axes[2, col].set_title(f"{s}: policy-driven flagged steps above the take threshold", fontsize=9)
        for ax, ylabel in zip(axes[:, col], ("share of attempts", "expert share", "share of steps")):
            ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(axis="x", labelsize=7); ax.legend(fontsize=7)
    return _finish(plt, fig, figures, f"dagger_{r}_flags.png", f"round {r}: failure flags during DAgger attempts")


# ---------------------------------------------------------------- takeovers and attempts


def takeover_stages_figure(plt, figures: Path, r: str, tos: dict[str, list[dict]]) -> str:
    """Per scorer, takeovers counted by the stage they started in (rows) and where they ended (columns): the stage
    the policy took back in, or the episode's end with the expert still driving. Shaded by the share of the row."""
    from matplotlib.colors import LinearSegmentedColormap

    # The expert's plan can run on after the task is done, so a hand-back can be in the last stage (the policy's next
    # step then registers the success); a takeover starting there is not expected (the policy's success ends the episode).
    starts = ends = stages.STAGES
    columns = list(ends) + ["success", "failure"]
    cmap = LinearSegmentedColormap.from_list("share", SHADES)
    n = len(starts)
    fig, axes = plt.subplots(1, len(tos), figsize=(max(8.0, 0.55 * (n + 2) + 2) * len(tos), max(4.8, 0.36 * n + 2.4)), squeeze=False)
    for ax, (s, rows) in zip(axes[0], tos.items()):
        counts = np.zeros((len(starts), len(columns)))
        for t in rows:
            if t["ended"] == "returned":
                j = ends.index(t["end_stage"])
            else:
                j = len(ends) + (0 if t["episode_success"] == "True" else 1)
            counts[starts.index(t["stage"]), j] += 1
        share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        ax.imshow(np.ma.masked_where(counts == 0, share), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, j in zip(*np.nonzero(counts)):
            ax.text(j, i, f"{int(counts[i, j])}\n{100 * share[i, j]:.0f}%", ha="center", va="center", fontsize=7 if n <= 8 else 6,
                    color="white" if share[i, j] > 0.45 else INK)
        ax.axvline(len(ends) - 0.5, color=INK_MUTED, lw=1.2)
        ax.text((len(ends) - 1) / 2, -0.8, "handed back to the policy in", ha="center", fontsize=8, color=INK_MUTED)
        ax.text(len(ends) + 0.5, -0.8, "expert drove to the end", ha="center", fontsize=8, color=INK_MUTED)
        _stage_ticks(ax, range(len(columns)), columns)
        ax.set_yticks(range(len(starts)), [f"{name} ({int(c)})" for name, c in zip(starts, counts.sum(1))], fontsize=8)
        ax.set_ylabel("stage the takeover started in (count)", fontsize=8)
        ax.set_title(f"{s}: {int(counts.sum())} takeovers (shade: share of the row)", fontsize=9, pad=28)
    return _finish(plt, fig, figures, f"dagger_{r}_takeover_stages.png", f"round {r}: where takeovers start and end")


def takeover_timing_figure(plt, figures: Path, r: str, eps: dict[str, list[dict]], tos: dict[str, list[dict]],
                           per_step: dict[str, list[dict]], colors: dict[str, str]) -> str:
    """When the expert takes over and hands back, the share of running attempts it drives at each step, and the stage
    of each attempt's first takeover (takeover counts and lengths per stage: ``takeovers_by_stage_figure``)."""
    fig, ((ax_start, ax_back), (ax_share, ax_first)) = plt.subplots(2, 2, figsize=(14, 7.4))
    longest = max([int(e["steps"]) for rows in eps.values() for e in rows]
                  + [len(e["controller"]) for rows in per_step.values() for e in rows] + [1])
    bins = np.arange(0, longest + 10, 10)
    firsts = _start_stages([[{"stage": e["first_take_stage"]} for e in rows] for rows in eps.values()]) + ["none"]
    width = 0.8 / len(tos)
    for k, s in enumerate(tos):
        color = colors.get(s)
        begin = [int(t["start"]) for t in tos[s]]
        ax_start.hist(begin, bins=bins, histtype="step", lw=2, color=color, label=f"{s} ({len(begin)} takeovers)")
        back = [int(t["end"]) for t in tos[s] if t["ended"] == "returned"]
        ax_back.hist(back, bins=bins, histtype="step", lw=2, color=color, label=f"{s} ({len(back)} hand-backs)")
        if per_step.get(s):
            running, expert = np.zeros(longest), np.zeros(longest)
            for e in per_step[s]:
                n = len(e["controller"])
                running[:n] += 1
                expert[:n] += e["controller"] == EXPERT
            keep = running >= max(1, 0.05 * len(per_step[s]))  # leave out the last few long attempts
            ax_share.plot(np.flatnonzero(keep), (expert / np.maximum(running, 1))[keep], color=color, lw=2, label=s)
        first = [e["first_take_stage"] or "none" for e in eps[s]]
        ax_first.bar(np.arange(len(firsts)) + k * width, [first.count(c) / max(1, len(first)) for c in firsts], width,
                     color=color, label=f"{s} ({len(first)} attempts)")
    for ax, title, ylabel in ((ax_start, "step the expert took over", "takeovers"),
                              (ax_back, "step the expert handed back to the policy", "hand-backs"),
                              (ax_share, "share of running attempts the expert drives", "share")):
        ax.set_title(title, fontsize=9); ax.set_xlabel("step", fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(0, longest); ax.legend(fontsize=7)
    ax_share.set_ylim(0, 1)
    _stage_ticks(ax_first, np.arange(len(firsts)) + width * (len(tos) - 1) / 2, firsts)
    ax_first.set_title("stage of an attempt's first takeover", fontsize=9); ax_first.set_ylabel("share of attempts", fontsize=8)
    ax_first.legend(fontsize=7)
    return _finish(plt, fig, figures, f"dagger_{r}_takeover_timing.png", f"round {r}: when the expert takes over")


def episodes_figure(plt, figures: Path, r: str, per_step: dict[str, list[dict]], shown: tuple[int, int] = (40, 20)) -> str:
    """Sampled attempts step by step, per scorer: up to 40 successful ones (spread over the seeds) above up to 20
    failed ones, each group sorted by the first takeover. Color: the task stage; black line: the expert drives."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colors = stage_colors()
    palette = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in colors])

    def first_takeover(e: dict) -> int:
        return int(np.argmax(e["controller"] == EXPERT)) if (e["controller"] == EXPERT).any() else 10**6

    groups: dict[str, list] = {}
    for s, rows in per_step.items():
        groups[s] = []
        for success, k in zip((True, False), shown):
            pool = [e for e in rows if e["success"] == success]
            idx = sorted(set(np.linspace(0, len(pool) - 1, min(k, len(pool))).round().astype(int))) if pool else []
            groups[s].append((success, len(pool), sorted((pool[i] for i in idx), key=first_takeover)))
    heights = [max([1] + [len(groups[s][g][2]) for s in groups]) for g in (0, 1)]
    longest = max(len(e["controller"]) for rows in per_step.values() for e in rows)
    fig, axes = plt.subplots(2, len(per_step), figsize=(7.4 * len(per_step), 1.8 + 0.16 * sum(heights)), sharex=True,
                             gridspec_kw=dict(height_ratios=heights), squeeze=False)
    for col, s in enumerate(per_step):
        for row, (success, total, pick) in enumerate(groups[s]):
            ax, n = axes[row, col], max(1, len(pick))
            img = np.ones((4 * n, longest, 3))  # 3 pixel rows of stage color and a white gap per attempt
            for i, e in enumerate(pick):
                img[4 * i:4 * i + 3, :len(e["stage"])] = palette[e["stage"]]
            ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, longest, n, 0))
            for i, e in enumerate(pick):
                for start, length in _runs(e["controller"] == EXPERT):
                    ax.hlines(i + 0.375, start, start + length, color=INK, lw=1.6)
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{s}: {'successful' if success else 'failed'} attempts, {len(pick)} of {total} shown", fontsize=9)
        axes[-1, col].set_xlabel("step", fontsize=8)
    handles = [Patch(color=c, label=n) for c, n in zip(colors, stages.STAGES)]
    handles.append(Line2D([], [], color=INK, lw=2, label="expert driving"))
    rows = -(-len(handles) // 8)
    top = 1 - (0.5 + 0.25 * rows) / fig.get_figheight()
    fig.suptitle(f"round {r}: attempts step by step, each group sorted by its first takeover", fontsize=10, y=0.998)
    fig.legend(handles=handles, loc="upper center", ncol=min(8, len(handles)), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1 - 0.3 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    name = f"dagger_{r}_episodes.png"
    fig.savefig(figures / name, dpi=90)
    plt.close(fig)
    return name


def takeovers_by_stage_figure(plt, figures: Path, collections: list[tuple[str, str, list[dict], list[dict]]]) -> str:
    """Box plots per task stage over every DAgger collection (one box each, side by side): the takeovers that started
    in the stage per attempt that spent a step there, and the length (expert steps) of each takeover by the stage it
    started in. Boxes p25–p75 split at the median, whiskers p5–p95, white square the mean."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    names = _start_stages([tos for *_, tos in collections])
    fig, (ax_count, ax_length) = plt.subplots(2, 1, figsize=(min(24, 5 + 0.9 * len(names) * len(collections)), 8.6))
    w = 0.8 / len(collections)
    for i, (label, color, rows, tos) in enumerate(collections):
        x = np.arange(len(names)) + (i - (len(collections) - 1) / 2) * w
        started: dict[int, list[int]] = {}
        for t in tos:
            started.setdefault(t["seed"], []).append(t["stage"])
        counts = [[started.get(a["seed"], []).count(k) for a in rows if (a["stage"] == k).any()] for k in range(len(names))]
        lengths = [[t["length"] for t in tos if t["stage"] == k] for k in range(len(names))]
        for ax, values, note in ((ax_count, counts, "mean"), (ax_length, lengths, "n")):
            q = [_quantiles(v) for v in values]
            keep = [k for k in range(len(names)) if q[k]["n"]]
            if not keep:
                continue
            parts = ax.bxp([dict(med=q[k]["median"], q1=q[k]["p25"], q3=q[k]["p75"], whislo=q[k]["p5"], whishi=q[k]["p95"],
                                 mean=q[k]["mean"], fliers=[]) for k in keep],
                           positions=x[keep], widths=w * 0.8, patch_artist=True, showmeans=True, showcaps=False,
                           meanprops=dict(marker="s", markerfacecolor="white", markeredgecolor=INK, markersize=4.5),
                           medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK_MUTED, lw=1.1))
            for patch in parts["boxes"]:
                patch.set_facecolor(color); patch.set_edgecolor(color)
            for k in keep:
                if note == "mean" and not q[k]["mean"]:
                    continue  # attempts were there, none was taken over
                text = f"{q[k]['mean']:.2f}" if note == "mean" else f"{q[k]['n']}"
                ax.text(x[k], q[k]["p95"], text, ha="center", va="bottom", fontsize=6.5, color=INK_MUTED)
    ax_count.set_ylabel("takeovers per attempt", fontsize=8)
    ax_count.set_title("takeovers that started in the stage, per attempt that spent a step there (number above: the mean)",
                       fontsize=9)
    ax_length.set_ylabel("expert steps", fontsize=8)
    ax_length.set_title("length of each takeover, by the stage it started in (number above: takeovers)", fontsize=9)
    for ax in (ax_count, ax_length):
        _stage_ticks(ax, range(len(names)), names)
        ax.set_xlim(-0.6, len(names) - 0.4); ax.set_ylim(bottom=0); ax.margins(y=0.12)
        for k in range(len(names) - 1):
            ax.axvline(k + 0.5, color=PALE, lw=0.8)
    handles = [Patch(color=color, label=f"{label} ({len(rows)} attempts, {len(tos)} takeovers)") for label, color, rows, tos in collections]
    handles.append(Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=INK, label="mean"))
    ax_count.legend(handles=handles, fontsize=7, loc="upper right")
    return _finish(plt, fig, figures, "dagger_takeovers_by_stage.png",
                   "takeovers by task stage over the DAgger collections (boxes p25–p75 split at the median, whiskers p5–p95)")


# ---------------------------------------------------------------- entry points


def collections(cfg: dict) -> list[tuple[str, str, list[dict], list[dict]]]:
    """(label, color, attempts, takeovers) of every collection with a ``_steps.npz``, round by round. The scorer sets
    the hue, the round its shade."""
    shades = {"vig": ("#3b6ea5", "#86b6ef"), "flow": ("#d1793b", "#f0a77f")}
    results, out = Paths.of(cfg).results(), []
    for i, r in enumerate(cfg["rounds"]):
        for s in cfg["dagger"]["scorers"]:
            rows, res = load_steps(results / f"dagger_{r}_{s}_steps.npz"), results / f"dagger_{r}_{s}.json"
            if rows and res.exists():
                color = shades.get(s, (INK_MUTED, PALE))[min(i, 1)]
                out.append((f"{r} · {s}", color, rows, takeovers(rows, json.loads(res.read_text())["settings"])))
    return out


def summary_figures(cfg: dict, plt, figures_dir: Path) -> list[str]:
    """The figures over every collection of every round."""
    found = collections(cfg)
    return [takeovers_by_stage_figure(plt, figures_dir, found)] if found else []


def round_data(cfg: dict, r: str) -> tuple[dict, dict]:
    """Per scorer with a ``_steps.npz``: its attempts, settings, takeovers, stage and flag tables; and the round's
    demo-frame scores per scorer (empty if the file is not there)."""
    results, data = Paths.of(cfg).results(), {}
    for s in cfg["dagger"]["scorers"]:
        rows, res = load_steps(results / f"dagger_{r}_{s}_steps.npz"), results / f"dagger_{r}_{s}.json"
        if rows and res.exists():
            settings = json.loads(res.read_text())["settings"]
            data[s] = dict(rows=rows, settings=settings, takeovers=takeovers(rows, settings),
                           table=stage_stats(rows, settings), flags=flag_stats(rows, settings))
    path = Paths.of(cfg).demo_scores(r)
    demo = {}
    if path.exists():
        z = np.load(path)
        demo = {s: np.asarray(z[s], float) for s in cfg["dagger"]["scorers"] if s in z}
    return data, demo


def figures(cfg: dict, plt, figures_dir: Path, r: str, eps: dict, tos: dict, colors: dict) -> list[str]:
    """Every figure of this module for round ``r`` (those whose inputs exist), and the per-stage CSVs."""
    data, demo = round_data(cfg, r)
    results, made = Paths.of(cfg).results(), []
    for s, d in data.items():
        _write_csv(results / f"dagger_{r}_{s}_by_stage.csv", d["table"])
        _write_csv(results / f"dagger_{r}_{s}_by_flag.csv", d["flags"])
    if data:
        made += [stage_outcomes_figure(plt, figures_dir, r, data), stage_scores_figure(plt, figures_dir, r, data, demo),
                 stage_take_figure(plt, figures_dir, r, data, demo), stage_return_figure(plt, figures_dir, r, data, demo),
                 flags_figure(plt, figures_dir, r, data)]
    scorers = [s for s in data if tos.get(s) and "end_stage" in tos[s][0]]
    if scorers:
        made += [takeover_stages_figure(plt, figures_dir, r, {s: tos[s] for s in scorers}),
                 takeover_timing_figure(plt, figures_dir, r, {s: eps[s] for s in scorers}, {s: tos[s] for s in scorers},
                                        {s: data[s]["rows"] for s in scorers}, colors),
                 episodes_figure(plt, figures_dir, r, {s: data[s]["rows"] for s in scorers})]
    return made
