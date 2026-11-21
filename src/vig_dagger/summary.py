"""Every collection side by side: the experiment's numbers per collection, and per setting over the MoE seeds.

Per collection (``results/collections/<collection>.json``, written by ``analysis.py``):

  1. collection success rate     successful episodes / all attempts (failed attempts counted)
  2. expert share of the data    expert-driven steps / all steps of the collection's DAgger data (the first
                                 ``dagger.num_demos`` successful episodes); the random move's share separately
  3. expert share of all         the same over every attempt, successful or failed
     attempts
  4. expert share of the failed  the same over the failed attempts, and the expert steps per failed attempt
     attempts

each pooled over steps, plus the per-episode medians. Per setting (scorer, take-over and hand-back threshold,
variant): the mean and standard deviation over the MoE seeds whose collection is finished and analyzed (a collection
analyzed mid-run is listed per collection but left out of the means), and each seed's value. Matching reference
collections (``reference.py``) get rows of their own (``reference: true``).

Outputs: ``results/summary.json`` (rows per collection and per setting, and the references), ``results/summary.csv``
(one row per collection) and ``results/figures/collections.png`` (the four numbers per setting as horizontal bars,
random-move steps stacked in orange on the expert shares; references below a line). Works on partial results too.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from vig_dagger import reference
from vig_dagger.config import Collection, Paths, collections

SUCCESS, EXPERT, MOVE, INK, INK2, MUTED, GRID = "#1baf7a", "#2a78d6", "#eb6834", "#1f1f1f", "#52514e", "#7a7873", "#e3e1dc"
METRICS = ("success_rate", "data_expert", "data_move", "all_expert", "all_move", "failed_expert", "failed_move",
           "failed_expert_frames", "failed_move_frames")


def _median(pe: dict, key: str):
    return (pe.get(key) or {}).get("median")


def row(r: dict, c: Collection, **extra) -> dict:
    """One collection's row from its analysis result (``results/collections/<id>.json``)."""
    m, st = r["metrics"], r["settings"]
    data, failed, everything = m["data"], m["failed"], m["all"]
    dp, fp, ap = data.get("pooled", {}), failed.get("pooled", {}), everything.get("pooled", {})
    dpe, fpe, ape = data.get("per_episode", {}), failed.get("per_episode", {}), everything.get("per_episode", {})
    n = m["collection"]
    nf = failed.get("episodes", 0)
    return dict(
        collection=c.id, moe_seed=c.seed, scorer=c.scorer, threshold=c.threshold, return_threshold=c.return_spec,
        variant=c.variant, finished=n["successes"] >= st["target"] or n["attempts"] >= st["max_attempts"],
        threshold_value=st["take_threshold"], return_threshold_value=st.get("return_threshold", st["take_threshold"]),
        attempts=n["attempts"], successes=n["successes"],
        data_episodes=data.get("episodes", 0), failed_episodes=nf, success_rate=n["success_rate"],
        data_expert=dp.get("expert_share"), data_move=dp.get("move_share"),
        data_expert_median=_median(dpe, "expert_share"), data_move_median=_median(dpe, "move_share"),
        all_expert=ap.get("expert_share"), all_move=ap.get("move_share"),
        all_expert_median=_median(ape, "expert_share"), all_move_median=_median(ape, "move_share"),
        failed_expert=fp.get("expert_share"), failed_move=fp.get("move_share"),
        failed_expert_median=_median(fpe, "expert_share"), failed_move_median=_median(fpe, "move_share"),
        failed_expert_frames=fp["expert_steps"] / nf if nf and "expert_steps" in fp else None,
        failed_move_frames=fp["move_steps"] / nf if nf and "move_steps" in fp else None,
        takeovers_median=_median(ape, "takeovers"), returns_median=_median(ape, "returns"),
        takeovers_mean=(ape.get("takeovers") or {}).get("mean"),
        random_moves=ap.get("random_moves"), moves_skipped=ap.get("moves_skipped"),
        reasons=failed.get("reasons", {}), **extra)


def collection_rows(cfg: dict) -> list[dict]:
    """One row per analyzed collection, in the experiment's collection order."""
    paths, rows = Paths.of(cfg), []
    for c in collections(cfg):
        path = paths.collection_results(c).with_suffix(".json")
        if path.is_file():
            rows.append(row(json.loads(path.read_text()), c, reference=False))
    return rows


def reference_rows(cfg: dict) -> tuple[list[dict], list[dict]]:
    """(rows of the matching references, the references that were excluded with their reasons)."""
    rows, excluded = [], []
    for ref in reference.load(cfg):
        if ref["problems"]:
            excluded.append(dict(id=ref["id"], dir=str(ref["dir"]), problems=ref["problems"]))
            continue
        rows.append(row(ref["result"], Collection.parse(ref["id"]), reference=True, reference_dir=str(ref["dir"])))
    return rows, excluded


def setting_key(r: dict) -> tuple:
    return r["scorer"], r["variant"], r["threshold"], r["return_threshold"]


def settings_rows(cfg: dict, rows: list[dict]) -> list[dict]:
    """One row per setting (scorer, variant, take-over and hand-back threshold): mean, std and per-seed values over the
    MoE seeds."""
    out, seen = [], []
    for c in collections(cfg):
        key = (c.scorer, c.variant, c.threshold, c.return_spec)
        if key not in seen:
            seen.append(key)
    for key in seen:
        scorer, variant, spec, back = key
        own = [r for r in rows if setting_key(r) == key and r["finished"]]
        s = dict(scorer=scorer, variant=variant, threshold=spec, return_threshold=back, seeds=[r["moe_seed"] for r in own],
                 attempts=sum(r["attempts"] for r in own), successes=sum(r["successes"] for r in own),
                 failed_episodes=sum(r["failed_episodes"] for r in own))
        for k in METRICS + ("data_expert_median", "all_expert_median", "failed_expert_median"):
            values = [r[k] for r in own if r[k] is not None]
            s[k] = dict(mean=float(np.mean(values)) if values else None,
                        std=float(np.std(values)) if len(values) > 1 else None,
                        by_seed={str(r["moe_seed"]): r[k] for r in own})
        out.append(s)
    return out


def summarize(cfg: dict) -> dict:
    paths = Paths.of(cfg)
    rows = collection_rows(cfg)
    settings = settings_rows(cfg, rows)
    refs, excluded = reference_rows(cfg)
    paths.results().mkdir(parents=True, exist_ok=True)
    out = dict(experiment=cfg["name"], collections_total=len(collections(cfg)), collections_analyzed=len(rows),
               collections_finished=sum(r["finished"] for r in rows),
               num_demos=int(cfg["dagger"]["num_demos"]), moe_seeds=[int(s) for s in cfg["moe"]["seeds"]],
               collections=rows, settings=settings, references=refs, references_excluded=excluded)
    (paths.results() / "summary.json").write_text(json.dumps(out, indent=1))
    if rows or refs:
        with open(paths.results() / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[k for k in (rows or refs)[0] if k != "reasons"] + ["reference_dir"],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows + refs)
    figure(cfg, settings, refs, paths.results() / "figures" / "collections.png")
    print(f"[summary] {len(rows)} of {len(collections(cfg))} collections analyzed, {len(refs)} reference(s) "
          f"({len(excluded)} excluded) -> {paths.results() / 'summary.json'}", flush=True)
    return out


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def setting_label(scorer: str, variant: str, spec: str, back: str, cfg: dict) -> str:
    d = cfg["dagger"]
    many_scorers, many_variants = len(d["scorers"]) > 1, len(d["variants"]) > 1
    text = f"hand-back < {back}" if back != spec else f"hand-back < {spec} (= take-over)"
    if len(d["thresholds"]) > 1:
        text = f"take-over > {spec} · " + text
    if many_scorers:
        text = ("VIG" if scorer == "vig" else "flow") + " · " + text
    if many_variants:
        text += " · random move" if variant == "random_move" else " · no random move"
    return text


def figure(cfg: dict, settings: list[dict], refs: list[dict], path) -> None:
    """The four numbers per setting as horizontal bars, one row per setting (mean over the MoE seeds; dots: each seed
    when there are several) and one per matching reference, below a line: success rate; expert share of the DAgger
    data, of all attempts and of the failed attempts, the random move's share stacked in orange."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = cfg["dagger"]
    n_seeds, n_demos = len(cfg["moe"]["seeds"]), int(d["num_demos"])
    entries = []  # (label, values per metric: (mean, per-seed list), n successes/attempts text, failed text, is_ref)
    for st in settings:
        label = setting_label(st["scorer"], st["variant"], st["threshold"], st["return_threshold"], cfg)
        entries.append(dict(label=label, s=st, ref=False))
    for r in refs:
        st = dict(seeds=[r["moe_seed"]], attempts=r["attempts"], successes=r["successes"], failed_episodes=r["failed_episodes"],
                  **{k: dict(mean=r[k], std=None, by_seed={str(r["moe_seed"]): r[k]}) for k in METRICS})
        label = setting_label(r["scorer"], r["variant"], r["threshold"], r["return_threshold"], cfg) + "  [reference]"
        entries.append(dict(label=label, s=st, ref=True))
    n = len(entries)
    panels = [("success_rate", None, "Success rate per attempt", "Successes / all attempts", 1.45),
              ("data_expert", "data_move", "Expert share of the DAgger data",
               f"Steps of the {n_demos} kept successful episodes", 1.25),
              ("all_expert", "all_move", "Expert share of all attempts", "Steps of every attempt, successful or failed", 1.25),
              ("failed_expert", "failed_move", "Expert share of the failed attempts",
               "Steps of the failed attempts; grey: how many, steps each", 1.85)]
    fig_h = 1.75 + 0.55 * n + 1.3
    fig, axes = plt.subplots(1, 4, figsize=(20, fig_h), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.1, 1.1, 1.6]))
    fig.subplots_adjust(left=0.18, right=0.99, top=1 - 1.75 / fig_h, bottom=1.25 / fig_h, wspace=0.12)
    ys = list(range(n - 1, -1, -1))
    split = sum(not e["ref"] for e in entries)
    for ax, (key, move_key, title, subtitle, xmax) in zip(axes, panels):
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ticks = [0, 0.25, 0.5, 0.75, 1.0]
        ax.set_xticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        ax.grid(axis="x", color=GRID, zorder=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0, colors=MUTED, labelsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
        if 0 < split < n:
            ax.axhline(n - split - 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        for y, e in zip(ys, entries):
            st = e["s"]
            mean = st[key]["mean"]
            if mean is None:
                ax.text(0.01, y, "not finished yet" if not e["ref"] else "–", va="center", fontsize=8.5, color=MUTED)
                continue
            if move_key is None:
                ax.barh(y, mean, 0.56, color=SUCCESS, zorder=2)
                bold, gray, total = _pct(mean), f"{st['successes']}/{st['attempts']}", mean
                values = [v for v in st[key]["by_seed"].values() if v is not None]
            else:
                move = st[move_key]["mean"] or 0.0
                ax.barh(y, mean, 0.56, color=EXPERT, zorder=2)
                if move:
                    ax.barh(y, move, 0.56, left=mean, color=MOVE, zorder=2, edgecolor="white", linewidth=1.2)
                bold = f"{_pct(mean)} + {_pct(move)}" if move else _pct(mean)
                total = mean + move
                values = [v + (st[move_key]["by_seed"].get(k) or 0.0) for k, v in st[key]["by_seed"].items() if v is not None]
                gray = ""
                if key == "failed_expert":
                    frames = st["failed_expert_frames"]["mean"]
                    mframes = st["failed_move_frames"]["mean"] or 0.0
                    gray = (f"{st['failed_episodes']} failed · {frames + mframes:.0f} steps each"
                            if frames is not None else f"{st['failed_episodes']} failed")
            if len(values) > 1:
                ax.scatter(values, [y] * len(values), s=12, color=INK, zorder=3, linewidths=0)
            x = max([total] + values)
            t = ax.annotate(bold, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9.5,
                            fontweight="bold", color=INK)
            if gray:
                fig.canvas.draw_idle()
                w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
                ax.annotate(gray, (x, y), xytext=(5 + w + 5, 0), textcoords="offset points", va="center", fontsize=8.3,
                            color=MUTED)
    axes[0].set_yticks(ys, [e["label"] for e in entries], fontsize=10)
    for tl, e in zip(axes[0].get_yticklabels(), entries):
        if e["ref"]:
            tl.set_color(INK2)
    axes[1].legend(handles=[Patch(color=EXPERT, label="expert"),
                            Patch(color=MOVE, label="random move (removed before retraining)")],
                   loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, fontsize=8.8, frameon=False)
    seeds = cfg["moe"]["seeds"]
    take = ", ".join(d["thresholds"])
    title = (f"PlugCharger DAgger collections, 7-level MoE {'seeds' if len(seeds) > 1 else 'seed'} "
             f"{', '.join(str(s) for s in seeds)}, expert '{d['expert']}': expert takes over after {d['take_count']} "
             f"scores above {take}, the policy takes back control after {d['return_count']} scores below the hand-back "
             "threshold")
    move = d["random_move"]
    note = (f"Random move: {move['radius'] * 100:g} cm / {move['rot_deg']:g}° at every take-over, before the expert "
            "(shown in orange; cut out of the retraining data). Shares pooled over steps; the policy drove the rest. "
            f"{n_demos} successes per collection." + (" Bars: mean over the MoE seeds; dots: each seed." if n_seeds > 1 else "")
            + (" [reference]: a collection made earlier with the same MoE, expert and settings except the hand-back "
               "threshold (reference.py checks this)." if refs else ""))
    fig.text(0.01, 1 - 0.4 / fig_h, textwrap.fill(title, 185, break_on_hyphens=False), fontsize=12, fontweight="bold",
             color=INK, va="top")
    fig.text(0.01, 0.12 / fig_h, textwrap.fill(note, 240, break_on_hyphens=False), fontsize=8.6, color=MUTED, va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfa")
    plt.close(fig)
