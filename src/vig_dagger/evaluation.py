"""Evaluation summary (part 4): every retrained policy, and the MoEs, on the same scenes.

Evaluated policies (``eval/<label>/level1/seeds_<first>.jsonl``, one line per episode, ``workers/evaluate.py``):

  retrain/<collection>/seed<s>   a dense full-resolution policy trained on the base demos + that collection's DAgger data
  moe/sd<s>                      the MoE that drove the collections of seed s, at full resolution

Per policy: success rate, steps of successful episodes, where episodes end (furthest stage, the stage a failure
ended in) and the failure flags. Per retrained policy: a paired comparison with its collection's driving MoE on the
same scenes (scenes only the retrained policy solved, scenes only the MoE solved, two-sided sign test on those). Per
setting (scorer, variant, threshold): the retrained policies' success over the MoE seeds (each collection's success
averaged over its training seeds; mean, std and each MoE seed), and the paired comparison summed over the seeds.

Outputs: ``results/eval_summary.json``, ``results/eval_episodes.csv`` (every episode) and
``results/figures/eval_success.png``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vig_dagger import stages
from vig_dagger.config import Paths, collections


def load(eval_dir: Path) -> dict[str, list[dict]]:
    """{policy label: episode records at full resolution} from ``eval/<label>/level1/*.jsonl``."""
    out: dict[str, list[dict]] = {}
    for path in sorted(eval_dir.glob("**/level1/*.jsonl")):
        label = str(path.parent.parent.relative_to(eval_dir))
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out.setdefault(label, []).extend(rows)
    return {k: sorted({r["seed"]: r for r in v}.values(), key=lambda r: r["seed"]) for k, v in out.items()}


def policy_metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    failed = [r for r in rows if not r["success"]]
    return dict(
        episodes=n, successes=len(succ), success_rate=len(succ) / n if n else None,
        success_steps_median=float(np.median([r["steps"] for r in succ])) if succ else None,
        failed_at={s: sum(r["failed_at"] == s for r in failed) for s in stages.STAGES},
        flags={k: float(np.mean([r[f"flag_{k}"] for r in rows])) if rows else None for k in stages.FLAGS})


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test (binomial, p = 1/2) on the discordant scenes."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _paired(rows: list[dict], ref: list[dict]) -> tuple[int, int]:
    """(scenes only rows solved, scenes only ref solved), over the scenes both were evaluated on."""
    other = {r["seed"]: r["success"] for r in ref}
    wins = sum(r["success"] and not other[r["seed"]] for r in rows if r["seed"] in other)
    losses = sum(other[r["seed"]] and not r["success"] for r in rows if r["seed"] in other)
    return int(wins), int(losses)


def summarize(cfg: dict) -> dict[str, Any]:
    paths = Paths.of(cfg)
    data = load(paths.root / "eval")
    n_eval = int(cfg["evaluation"]["num_episodes"])
    complete = {label: rows for label, rows in data.items() if len(rows) >= n_eval}  # only finished evaluations count
    moes = {int(s): complete.get(f"moe/sd{int(s)}") for s in cfg["moe"]["seeds"]}
    policies = {label: dict(label=label, complete=label in complete, **policy_metrics(rows)) for label, rows in sorted(data.items())}
    retrained = []
    for c in collections(cfg):
        for s in cfg["retrain"]["seeds"]:
            label = f"retrain/{c.id}/seed{int(s)}"
            rows = complete.get(label)
            if rows is None:
                continue
            row = dict(label=label, collection=c.id, moe_seed=c.seed, scorer=c.scorer, variant=c.variant,
                       threshold=c.threshold, train_seed=int(s), episodes=len(rows),
                       success_rate=float(np.mean([r["success"] for r in rows])))
            if moes.get(c.seed) is not None:
                wins, losses = _paired(rows, moes[c.seed])
                row.update(moe_success_rate=float(np.mean([r["success"] for r in moes[c.seed]])), only_this=wins,
                           only_moe=losses, sign_test_p=sign_test(wins, losses))
            retrained.append(row)
    d = cfg["dagger"]
    settings = []
    for scorer in d["scorers"]:
        for variant in d["variants"]:
            for spec in d["thresholds"]:
                own = [r for r in retrained if (r["scorer"], r["variant"], r["threshold"]) == (scorer, variant, spec)]
                per_moe: dict[int, list[float]] = {}
                for r in own:
                    per_moe.setdefault(r["moe_seed"], []).append(r["success_rate"])
                rates = {seed: float(np.mean(v)) for seed, v in sorted(per_moe.items())}
                wins = sum(r.get("only_this", 0) for r in own)
                losses = sum(r.get("only_moe", 0) for r in own)
                settings.append(dict(
                    scorer=scorer, variant=variant, threshold=spec, moe_seeds=sorted(rates),
                    success_by_moe_seed={str(k): v for k, v in rates.items()},
                    success_mean=float(np.mean(list(rates.values()))) if rates else None,
                    success_std=float(np.std(list(rates.values()))) if len(rates) > 1 else None,
                    only_this=wins, only_moe=losses, sign_test_p=sign_test(wins, losses) if own else None))
    moe_rates = {str(s): float(np.mean([r["success"] for r in rows])) for s, rows in moes.items() if rows is not None}
    out = dict(episodes_per_policy=n_eval, scenes=[int(cfg["evaluation"]["seed_start"]), int(cfg["evaluation"]["seed_start"]) + n_eval - 1],
               moes=moe_rates, retrained=retrained, settings=settings, policies=policies)
    results = paths.results()
    results.mkdir(parents=True, exist_ok=True)
    (results / "eval_summary.json").write_text(json.dumps(out, indent=1))
    rows_out = [dict(policy=label, **{k: v for k, v in r.items() if not isinstance(v, dict)}) for label, rows in sorted(data.items()) for r in rows]
    if rows_out:
        with open(results / "eval_episodes.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_out)
    if retrained or moe_rates:
        figure(cfg, out, results / "figures" / "eval_success.png")
    print(f"[eval] {len(retrained)} retrained policies and {len(moe_rates)} MoEs evaluated "
          f"({len(data) - len(complete)} evaluations still running)", flush=True)
    return out


def figure(cfg: dict, summary: dict, path: Path) -> None:
    """Success of the retrained policies per setting (bar: mean over the MoE seeds, dots: each seed), with the MoEs'
    own success at full resolution as a dashed line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vig_dagger.summary import GRID, INK, MUTED, SUCCESS

    d = cfg["dagger"]
    scorers, variants, specs = list(d["scorers"]), list(d["variants"]), list(d["thresholds"])
    per_scorer = len(variants) * len(specs) + 0.5 * (len(variants) - 1)
    by_key = {(s["scorer"], s["variant"], s["threshold"]): s for s in summary["settings"]}
    n_seeds = len(cfg["moe"]["seeds"])
    fig, ax = plt.subplots(figsize=(max(9.0, 0.95 * len(scorers) * len(variants) * len(specs) + 2), 5.6))
    ticks, groups = [], []
    for i, scorer in enumerate(scorers):
        x0 = i * (per_scorer + 1.5)
        for j, variant in enumerate(variants):
            for k, spec in enumerate(specs):
                x = x0 + j * (len(specs) + 0.5) + k
                ticks.append((x, spec + ("\n+ random\nmove" if variant == "random_move" else "")))
                s = by_key.get((scorer, variant, spec))
                if not s or s["success_mean"] is None:
                    ax.text(x, 0.01, "–", ha="center", va="bottom", color=MUTED, fontsize=10)
                    continue
                values = list(s["success_by_moe_seed"].values())
                ax.bar(x, s["success_mean"], 0.62, color=SUCCESS, zorder=2)
                ax.scatter([x] * len(values), values, s=14, color=INK, zorder=3, linewidths=0)
                label = f"{100 * s['success_mean']:.0f}%" + ("" if len(values) == n_seeds else f"\n(n={len(values)})")
                ax.annotate(label, (x, max([s["success_mean"]] + values)), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
        groups.append((x0 + (per_scorer - 1) / 2, "VIG" if scorer == "vig" else "flow loss"))
    moes = summary["moes"]
    if moes:
        from matplotlib.lines import Line2D

        mean = float(np.mean(list(moes.values())))
        ax.axhline(mean, color=INK, lw=1, ls="--", zorder=1)
        ax.legend(handles=[Line2D([], [], color=INK, lw=1, ls="--", label=f"driving MoEs at full resolution, mean {100 * mean:.0f}% ("
                                  + ", ".join(f"seed {k} {100 * v:.0f}%" for k, v in sorted(moes.items())) + ")"),
                           Line2D([], [], marker="o", ls="", color=INK, markersize=4, label="each MoE seed's collection")],
                  loc="upper right", fontsize=8.5, frameon=False)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=9)
    ax.set_xlim(min(t[0] for t in ticks) - 0.8, max(t[0] for t in ticks) + 0.8)
    ax.grid(axis="y", color=GRID, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=MUTED)
    ax.tick_params(axis="x", length=0)
    for gx, name in groups:
        ax.annotate(name, (gx, 0), xytext=(0, -52), textcoords="offset points", xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=11, fontweight="bold")
    n = summary["episodes_per_policy"]
    ax.set_title("Success of the retrained policies\n", loc="left", fontsize=13, fontweight="bold")
    ax.text(0, 1.02, f"Dense full-resolution policy on {cfg['demos']['num_demos']} base demos + the collection's "
            f"{d['num_demos']} DAgger episodes, {n} paired scenes. Bar: mean over the MoE seeds.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="#fcfcfa")
    plt.close(fig)

