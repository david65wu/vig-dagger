"""The report of an iteration-mode experiment (``iterate.rounds``): ``results/REPORT.md``, ``results/iterate_summary.json``
and the figures ``results/figures/thresholds.png`` (the adaptive threshold attempt by attempt, round after round) and
``results/figures/rounds.png`` (per round: the collection's success rate and expert share, the dense policy retrained
after the round and that round's MoE on the evaluation scenes).

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
from vig_dagger.config import Paths, collections, target_of
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
    out, previous_dense = [], None
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
        dense = complete.get(f"retrain/{col.id}/seed{int(cfg['retrain']['seeds'][0])}")
        moe = complete.get(f"moe/{col.moe}")
        for name, rows_ in (("dense", dense), ("moe", moe)):
            if rows_ is not None:
                k = int(sum(e["success"] for e in rows_))
                row[f"{name}_successes"], row[f"{name}_episodes"] = k, len(rows_)
                row[f"{name}_success_rate"] = k / len(rows_)
        if dense is not None and moe is not None:
            w, lo = _paired(dense, moe)
            row["dense_vs_moe"] = dict(only_dense=w, only_moe=lo, p=sign_test(w, lo))
        if dense is not None and previous_dense is not None:
            w, lo = _paired(dense, previous_dense)
            row["dense_vs_previous"] = dict(only_this=w, only_previous=lo, p=sign_test(w, lo))
        previous_dense = dense
        out.append(row)
    return out


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


def figure_rounds(rows: list[dict], path: Path) -> bool:
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
    for key, label, color in (("dense", "dense policy retrained after the round", "#1baf7a"),
                              ("moe", "the round's MoE", "#8e44ad")):
        ks = [(r["round"], r.get(f"{key}_successes"), r.get(f"{key}_episodes")) for r in rows if r.get(f"{key}_episodes")]
        if not ks:
            continue
        xs = np.array([k[0] for k in ks])
        rate = np.array([100 * k[1] / k[2] for k in ks])
        ci = np.array([[100 * v for v in wilson(k[1], k[2])] for k in ks])
        b.errorbar(xs, rate, yerr=[rate - ci[:, 0], ci[:, 1] - rate], marker="o", ms=6, lw=1.8, capsize=3, color=color, label=label)
    b.set_title("B · Success on the evaluation scenes (Wilson 95 %)", loc="left", fontsize=10, color=INK)
    b.set_ylabel("success (%)", color=MUTED, fontsize=9)
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xlabel("round", color=MUTED, fontsize=9)
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
    (results / "iterate_summary.json").write_text(json.dumps(summary, indent=1))
    made = [name for name, ok in (("thresholds.png", figure_thresholds(cfg, figures / "thresholds.png")),
                                  ("rounds.png", figure_rounds(rows, figures / "rounds.png"))) if ok]
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
    running = 0
    for r in rows:
        running += r["target"]
        dense = (f"{r['dense_successes']}/{r['dense_episodes']} = {_pct(r['dense_success_rate'])}"
                 if r.get("dense_episodes") else "–")
        moe = f"{r['moe_successes']}/{r['moe_episodes']} = {_pct(r['moe_success_rate'])}" if r.get("moe_episodes") else "–"
        lines.append(f"| {r['round']} | {cfg['demos']['num_demos']} + {running} | {dense} | "
                     f"{_test(r.get('dense_vs_previous'), 'only_this', 'only_previous')} | {moe} | "
                     f"{_test(r.get('dense_vs_moe'), 'only_dense', 'only_moe')} |")
    if made:
        lines += ["", "## Figures", ""] + [f"![{m}](figures/{m})" for m in made]
    (results / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary
