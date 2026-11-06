"""Every collection side by side: the experiment's three numbers per collection, and per setting over the MoE seeds.

Per collection (``results/collections/<collection>.json``, written by ``analysis.py``):

  1. collection success rate     successful episodes / all attempts (failed attempts counted)
  2. expert share of the data    expert-driven steps / all steps of the collection's DAgger data (the first
                                 ``dagger.num_demos`` successful episodes); the random move's share separately
  3. expert share of the failed  the same over the failed attempts
     attempts

each pooled over steps, plus the per-episode medians. Per setting (scorer, threshold, variant): the mean and standard
deviation over the MoE seeds whose collection is finished and analyzed (a collection analyzed mid-run is listed per
collection but left out of the means), and each seed's value.

Outputs: ``results/summary.json`` (rows per collection and per setting), ``results/summary.csv`` (one row per
collection) and ``results/figures/collections.png`` (the three numbers per setting: bars = mean over the MoE seeds,
dots = each seed, random-move steps stacked in orange on the expert share). Works on partial results too.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from vig_dagger.config import Paths, collections

SUCCESS, EXPERT, MOVE, INK, MUTED, GRID = "#1baf7a", "#2a78d6", "#eb6834", "#1f1f1f", "#7a7873", "#e3e1dc"
METRICS = ("success_rate", "data_expert", "data_move", "failed_expert", "failed_move")


def _median(pe: dict, key: str):
    return (pe.get(key) or {}).get("median")


def collection_rows(cfg: dict) -> list[dict]:
    """One row per analyzed collection, in the experiment's collection order."""
    paths, rows = Paths.of(cfg), []
    for c in collections(cfg):
        path = paths.collection_results(c).with_suffix(".json")
        if not path.is_file():
            continue
        r = json.loads(path.read_text())
        m, st = r["metrics"], r["settings"]
        data, failed, everything = m["data"], m["failed"], m["all"]
        dp, fp, ap = data.get("pooled", {}), failed.get("pooled", {}), everything.get("pooled", {})
        dpe, fpe, ape = data.get("per_episode", {}), failed.get("per_episode", {}), everything.get("per_episode", {})
        n = m["collection"]
        finished = n["successes"] >= st["target"] or n["attempts"] >= st["max_attempts"]
        rows.append(dict(
            collection=c.id, moe_seed=c.seed, scorer=c.scorer, threshold=c.threshold, variant=c.variant, finished=finished,
            threshold_value=st["take_threshold"], attempts=m["collection"]["attempts"], successes=m["collection"]["successes"],
            data_episodes=data.get("episodes", 0), failed_episodes=failed.get("episodes", 0),
            success_rate=m["collection"]["success_rate"],
            data_expert=dp.get("expert_share"), data_move=dp.get("move_share"),
            data_expert_median=_median(dpe, "expert_share"), data_move_median=_median(dpe, "move_share"),
            failed_expert=fp.get("expert_share"), failed_move=fp.get("move_share"),
            failed_expert_median=_median(fpe, "expert_share"), failed_move_median=_median(fpe, "move_share"),
            takeovers_median=_median(ape, "takeovers"), returns_median=_median(ape, "returns"),
            random_moves=ap.get("random_moves"), moves_skipped=ap.get("moves_skipped"),
            reasons=failed.get("reasons", {})))
    return rows


def settings_rows(cfg: dict, rows: list[dict]) -> list[dict]:
    """One row per setting (scorer, threshold, variant): mean, std and per-seed values over the MoE seeds."""
    out, d = [], cfg["dagger"]
    for scorer in d["scorers"]:
        for variant in d["variants"]:
            for spec in d["thresholds"]:
                own = [r for r in rows if (r["scorer"], r["variant"], r["threshold"]) == (scorer, variant, spec) and r["finished"]]
                row = dict(scorer=scorer, variant=variant, threshold=spec, seeds=[r["moe_seed"] for r in own],
                           attempts=sum(r["attempts"] for r in own), successes=sum(r["successes"] for r in own))
                for key in METRICS + ("data_expert_median", "failed_expert_median"):
                    values = [r[key] for r in own if r[key] is not None]
                    row[key] = dict(mean=float(np.mean(values)) if values else None,
                                    std=float(np.std(values)) if len(values) > 1 else None,
                                    by_seed={str(r["moe_seed"]): r[key] for r in own})
                out.append(row)
    return out


def summarize(cfg: dict) -> dict:
    paths = Paths.of(cfg)
    rows = collection_rows(cfg)
    settings = settings_rows(cfg, rows)
    paths.results().mkdir(parents=True, exist_ok=True)
    out = dict(experiment=cfg["name"], collections_total=len(collections(cfg)), collections_analyzed=len(rows),
               collections_finished=sum(r["finished"] for r in rows),
               num_demos=int(cfg["dagger"]["num_demos"]), moe_seeds=[int(s) for s in cfg["moe"]["seeds"]],
               collections=rows, settings=settings)
    (paths.results() / "summary.json").write_text(json.dumps(out, indent=1))
    if rows:
        with open(paths.results() / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "reasons"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    figure(cfg, settings, paths.results() / "figures" / "collections.png")
    print(f"[summary] {len(rows)} of {len(collections(cfg))} collections analyzed -> {paths.results() / 'summary.json'}",
          flush=True)
    return out


def _pct(x) -> str:
    return f"{100 * x:.0f}%"


def figure(cfg: dict, settings: list[dict], path) -> None:
    """The three numbers per setting, in three panels (success rate; expert share of the data; of the failed
    attempts): bars = mean over the MoE seeds, dots = each seed, the random move's share stacked in orange."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cfg["dagger"]
    scorers, variants, specs = list(d["scorers"]), list(d["variants"]), list(d["thresholds"])
    per_scorer = len(variants) * len(specs) + 0.5 * (len(variants) - 1)
    xs, ticks, groups = {}, [], []
    for i, scorer in enumerate(scorers):
        x0 = i * (per_scorer + 1.5)
        for j, variant in enumerate(variants):
            for k, spec in enumerate(specs):
                x = x0 + j * (len(specs) + 0.5) + k
                xs[(scorer, variant, spec)] = x
                ticks.append((x, spec + ("\n+ random\nmove" if variant == "random_move" else "")))
        groups.append((x0 + (per_scorer - 1) / 2, "VIG" if scorer == "vig" else "flow loss"))
    by_key = {(s["scorer"], s["variant"], s["threshold"]): s for s in settings}
    n_seeds, n_demos = len(cfg["moe"]["seeds"]), int(d["num_demos"])

    fig, axes = plt.subplots(3, 1, figsize=(max(9.0, 0.95 * len(xs) + 2), 14.5))
    panels = [("success_rate", None, "Success rate per attempt",
               "Successful episodes / all attempts (failed attempts saved and counted)"),
              ("data_expert", "data_move", "Expert share of the DAgger data",
               f"Share of the steps in each collection's {n_demos} successful episodes; the policy drove the rest"),
              ("failed_expert", "failed_move", "Expert share of the failed attempts",
               "Share of the steps in the failed attempts (not training data); the policy drove the rest")]
    for ax, (key, move_key, title, subtitle) in zip(axes, panels):
        top = 0.0
        for (scorer, variant, spec), x in xs.items():
            s = by_key.get((scorer, variant, spec))
            mean = s[key]["mean"] if s else None
            if mean is None:
                ax.text(x, 0.01, "–", ha="center", va="bottom", color=MUTED, fontsize=10)
                continue
            if move_key is None:
                ax.bar(x, mean, 0.62, color=SUCCESS, zorder=2)
                label, total = _pct(mean), mean
                values = list(s[key]["by_seed"].values())
            else:
                move = s[move_key]["mean"] or 0.0
                ax.bar(x, mean, 0.62, color=EXPERT, zorder=2)
                if variant == "random_move":
                    ax.bar(x, move, 0.62, bottom=mean, color=MOVE, zorder=2, edgecolor="white", linewidth=0.8)
                    label = f"{_pct(mean)} + {_pct(move)}"
                else:
                    label = _pct(mean)
                total = mean + move
                values = [e + (s[move_key]["by_seed"].get(seed) or 0.0) for seed, e in s[key]["by_seed"].items() if e is not None]
            ax.scatter([x] * len(values), values, s=14, color=INK, zorder=3, linewidths=0)
            seeds_here = len(s["seeds"])
            label += "" if seeds_here == n_seeds else f"\n(n={seeds_here})"
            y = max([total] + values)
            ax.annotate(label, (x, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color=INK)
            if move_key is None:  # successes / attempts, summed over the seeds, above the label
                ax.annotate(f"{s['successes']}/{s['attempts']}", (x, y), xytext=(0, 5 + 12 * (label.count("\n") + 1)),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8, color=MUTED)
            top = max(top, total, *values)
        ax.set_ylim(0, max(0.3, min(1.15, top + 0.2)) if key != "success_rate" else 1.18)
        ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1e-9, 0.2 if ax.get_ylim()[1] > 0.6 else 0.1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=9)
        ax.set_xlim(min(xs.values()) - 0.8, max(xs.values()) + 0.8)
        ax.grid(axis="y", color=GRID, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0, colors=MUTED)
        ax.tick_params(axis="x", length=0)
        for gx, name in groups:
            ax.annotate(name, (gx, 0), xytext=(0, -52), textcoords="offset points", xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=11, fontweight="bold")
        ax.set_title(f"{title}\n", loc="left", fontsize=13, fontweight="bold")
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED)
        if move_key is not None:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            ax.legend(handles=[Patch(color=EXPERT, label="expert"), Patch(color=MOVE, label="random move"),
                               Line2D([], [], marker="o", ls="", color=INK, markersize=4, label="each MoE seed")],
                      loc="upper right", fontsize=8.5, frameon=False)
    import textwrap

    width = int(fig.get_figwidth() * 10.5)  # characters per line of the title and the footnote
    seeds = cfg["moe"]["seeds"]
    title = (f"PlugCharger DAgger collections, 7-level MoE {'seeds' if len(seeds) > 1 else 'seed'} "
             f"{', '.join(str(s) for s in seeds)}: one threshold for take-over and hand-back "
             f"({d['take_count']} scores above / {d['return_count']} below)")
    note = (("Bars: mean over the MoE seeds whose collection is finished"
             + (f" (n shown when fewer than {n_seeds})" if n_seeds > 1 else "") + "; dots: each seed. ")
            + f"Random move: {d['random_move']['radius'] * 100:g} cm / {d['random_move']['rot_deg']:g}° at every take-over, "
            f"before the expert. {n_demos} successes per collection.")
    fig.suptitle(textwrap.fill(title, width), x=0.01, ha="left", fontsize=12, color="#3d3c39")
    fig.text(0.01, 0.005, textwrap.fill(note, int(width * 1.3)), fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 0.975), h_pad=4.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="#fcfcfa")
    plt.close(fig)
