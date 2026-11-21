"""Evaluation summary (part 4): every retrained policy, and the MoEs, on the same scenes.

Evaluated policies (``eval/<label>/level1/seeds_<first>.jsonl``, one line per episode, ``workers/evaluate.py``):

  retrain/<collection>/seed<s>   a dense full-resolution policy trained on the base demos + that collection's DAgger data
  moe/sd<s>                      the MoE that drove the collections of seed s, at full resolution

Per policy: success rate, steps of successful episodes, where episodes end (furthest stage, the stage a failure
ended in) and the failure flags. Per retrained policy: a paired comparison with its collection's driving MoE on the
same scenes (scenes only the retrained policy solved, scenes only the MoE solved, two-sided sign test on those). Per
setting (scorer, variant, take-over and hand-back threshold): the retrained policies' success over the MoE seeds (each
collection's success averaged over its training seeds; mean, std and each MoE seed), and the paired comparison summed
over the seeds. Matching references (``reference.py``): their retrained policy's episodes, paired with the MoE and
with each of this experiment's retrained policies.

Outputs: ``results/eval_summary.json``, ``results/eval_episodes.csv`` (every episode) and
``results/figures/retraining.png``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vig_dagger import reference, stages
from vig_dagger.config import Collection, Paths, collections


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
                       threshold=c.threshold, return_threshold=c.return_spec, train_seed=int(s), episodes=len(rows),
                       successes=int(sum(r["success"] for r in rows)), success_rate=float(np.mean([r["success"] for r in rows])))
            if moes.get(c.seed) is not None:
                wins, losses = _paired(rows, moes[c.seed])
                row.update(moe_success_rate=float(np.mean([r["success"] for r in moes[c.seed]])), only_this=wins,
                           only_moe=losses, sign_test_p=sign_test(wins, losses))
            retrained.append(row)
    refs, excluded = [], []
    first = int(cfg["evaluation"]["seed_start"])
    for ref in reference.load(cfg):
        rows = [r for r in ref.get("eval_rows", []) if first <= r["seed"] < first + n_eval]
        if ref["problems"] or len(rows) < n_eval:
            excluded.append(dict(id=ref["id"], problems=ref["problems"] or [f"its evaluation has {len(rows)} of the "
                                                                             f"{n_eval} scenes"]))
            continue
        c = Collection.parse(ref["id"])
        row = dict(label=f"reference/{c.id}", collection=c.id, moe_seed=c.seed, scorer=c.scorer, variant=c.variant,
                   threshold=c.threshold, return_threshold=c.return_spec, episodes=len(rows), reference=True,
                   successes=int(sum(r["success"] for r in rows)), success_rate=float(np.mean([r["success"] for r in rows])),
                   retrain=ref["retrain"], rows=rows)
        if moes.get(c.seed) is not None:
            wins, losses = _paired(rows, moes[c.seed])
            row.update(moe_success_rate=float(np.mean([r["success"] for r in moes[c.seed]])), only_this=wins,
                       only_moe=losses, sign_test_p=sign_test(wins, losses))
        for r in retrained:  # each of this experiment's retrained policies against the reference policy
            wins, losses = _paired(complete[r["label"]], rows)
            r.setdefault("vs_reference", {})[c.id] = dict(only_this=wins, only_reference=losses, sign_test_p=sign_test(wins, losses))
        refs.append(row)
    settings, seen = [], []
    for c in collections(cfg):
        key = (c.scorer, c.variant, c.threshold, c.return_spec)
        if key in seen:
            continue
        seen.append(key)
        own = [r for r in retrained if (r["scorer"], r["variant"], r["threshold"], r["return_threshold"]) == key]
        per_moe: dict[int, list[float]] = {}
        for r in own:
            per_moe.setdefault(r["moe_seed"], []).append(r["success_rate"])
        rates = {seed: float(np.mean(v)) for seed, v in sorted(per_moe.items())}
        wins = sum(r.get("only_this", 0) for r in own)
        losses = sum(r.get("only_moe", 0) for r in own)
        settings.append(dict(
            scorer=c.scorer, variant=c.variant, threshold=c.threshold, return_threshold=c.return_spec,
            moe_seeds=sorted(rates), success_by_moe_seed={str(k): v for k, v in rates.items()},
            successes=sum(r["successes"] for r in own), episodes=sum(r["episodes"] for r in own),
            success_mean=float(np.mean(list(rates.values()))) if rates else None,
            success_std=float(np.std(list(rates.values()))) if len(rates) > 1 else None,
            only_this=wins, only_moe=losses, sign_test_p=sign_test(wins, losses) if own else None))
    moe_rates = {str(s): float(np.mean([r["success"] for r in rows])) for s, rows in moes.items() if rows is not None}
    moe_counts = {str(s): [int(sum(r["success"] for r in rows)), len(rows)] for s, rows in moes.items() if rows is not None}
    out = dict(episodes_per_policy=n_eval, scenes=[first, first + n_eval - 1],
               moes=moe_rates, moe_counts=moe_counts, retrained=retrained, settings=settings,
               references=[{k: v for k, v in r.items() if k != "rows"} for r in refs], references_excluded=excluded,
               policies=policies)
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
        figure(cfg, out, results / "figures" / "retraining.png")
    print(f"[eval] {len(retrained)} retrained policies and {len(moe_rates)} MoEs evaluated "
          f"({len(data) - len(complete)} evaluations still running)", flush=True)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def figure(cfg: dict, summary: dict, path: Path) -> None:
    """Success of every evaluated policy on the same scenes, as horizontal bars with Wilson 95 % intervals: the MoEs
    (grey, the reference line), this experiment's retrained policies per setting (bar: mean over the MoE seeds; dots:
    each seed when there are several) and the matching references' retrained policies; paired sign tests in the
    footnote. Says that the random-move steps were cut out of the training data when they were."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vig_dagger.summary import INK, INK2, MUTED, GRID, SUCCESS, setting_label

    d, n = cfg["dagger"], summary["episodes_per_policy"]
    cut = bool(cfg["retrain"]["cut_random_moves"]) and "random_move" in d["variants"]
    rows = []  # (group, label, successes, episodes, per-seed rates, color)
    for s, (k, total) in sorted(summary.get("moe_counts", {}).items()):
        rows.append(("MoE (collected the DAgger data)", f"7-level MoE seed {s} at full resolution", k, total, [], MUTED))
    head = (f"Dense policy on {cfg['demos']['num_demos']} base demos + {d['num_demos']} DAgger episodes"
            + (", random-move steps REMOVED" if cut else ""))
    for st in summary["settings"]:
        if st["success_mean"] is None:
            rows.append((head, setting_label(st["scorer"], st["variant"], st["threshold"], st["return_threshold"], cfg),
                         None, None, [], SUCCESS))
            continue
        rates = list(st["success_by_moe_seed"].values())
        rows.append((head, setting_label(st["scorer"], st["variant"], st["threshold"], st["return_threshold"], cfg),
                     st["successes"], st["episodes"], rates if len(rates) > 1 else [], SUCCESS))
    for r in summary.get("references", []):
        rows.append(("Reference: collected earlier with the same MoE, expert and settings",
                     setting_label(r["scorer"], r["variant"], r["threshold"], r["return_threshold"], cfg) + "  [reference]",
                     r["successes"], r["episodes"], [], "#7fcfb0"))
    ypos, y, last = [], 0.0, None
    heads = []
    for g, *_ in rows:
        if g != last:
            y -= 1.25 if last is not None else 0.0
            heads.append((g, y))
            y -= 0.95
            last = g
        else:
            y -= 1.0
        ypos.append(y)
    lines = []
    fp = lambda x: "–" if x is None else f"{x:.2g}"  # no scene solved by only one of the two: no test
    for r in summary["retrained"]:
        name = setting_label(r["scorer"], r["variant"], r["threshold"], r["return_threshold"], cfg)
        if "only_this" in r:
            lines.append(f"{name} vs MoE: {r['only_this']} / {r['only_moe']}, p = {fp(r['sign_test_p'])}")
        for ref_id, t in (r.get("vs_reference") or {}).items():
            lines.append(f"{name} vs reference: {t['only_this']} / {t['only_reference']}, p = {fp(t['sign_test_p'])}")
    for r in summary.get("references", []):
        if "only_this" in r:
            lines.append(f"reference vs MoE: {r['only_this']} / {r['only_moe']}, p = {fp(r['sign_test_p'])}")
    notes = []
    if cut:
        notes.append("Random-move steps removed: every DAgger episode was cut at its random moves before training; the "
                     "move steps and the 15 before each (whose 16-step action chunk would contain the move) left "
                     "training, the expert's recovery after each move stayed in.")
    notes.append(f"Dense: one full-resolution policy (no experts), {cfg['training']['epochs']} epochs. Same {n} scenes for "
                 f"every policy (from {summary['scenes'][0]}), CPU physics, {cfg['evaluation']['max_episode_steps']} steps. "
                 "Whiskers: Wilson 95 % interval.")
    if lines:
        notes.append("Paired sign tests (exact, uncorrected), scenes only the first / only the second solved: "
                     + "; ".join(lines) + ".")
    note_text = "\n".join(textwrap.fill(x, 165) for x in notes)
    n_note = note_text.count("\n") + 1
    fig_h = 1.5 + 0.42 * (max(1.0, -min(ypos) + 1.3)) + 0.2 * n_note + 0.5
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.subplots_adjust(left=0.36, right=0.97, top=1 - 1.25 / fig_h, bottom=(0.35 + 0.2 * n_note) / fig_h)
    ax.set_ylim(min(ypos) - 0.7, 0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ["0%", "20%", "40%", "60%", "80%"])
    ax.grid(axis="x", color=GRID, zorder=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    for g, gy in heads:
        ax.text(-0.555, gy, g, transform=ax.get_yaxis_transform(), fontsize=10.5, fontweight="bold", color=INK, va="center")
    for (g, label, k, total, rates, color), yy in zip(rows, ypos):
        if k is None:
            ax.text(0.01, yy, "not evaluated yet", va="center", fontsize=8.5, color=MUTED)
            continue
        v = k / total
        lo, hi = wilson(k, total)
        ax.barh(yy, v, 0.56, color=color, zorder=2)
        ax.plot([lo, hi], [yy, yy], color=INK2, lw=1.1, zorder=4)
        for q in (lo, hi):
            ax.plot([q, q], [yy - 0.13, yy + 0.13], color=INK2, lw=1.1, zorder=4)
        if rates:
            ax.scatter(rates, [yy] * len(rates), s=12, color=INK, zorder=5, linewidths=0)
        ax.annotate(f"{100 * v:.1f}%", (hi, yy), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5,
                    fontweight="bold", color=INK)
        ax.annotate(f"{k}/{total}", (hi, yy), xytext=(48, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=MUTED)
    ax.set_yticks(ypos, [r[1] for r in rows], fontsize=9.5)
    title = "Success of the retrained policies" + (" (random-move steps removed from their training data)" if cut else "")
    fig.text(0.015, 1 - 0.42 / fig_h, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 1 - 0.78 / fig_h, f"Same {n} scenes for every policy; whiskers: Wilson 95 % interval.", fontsize=10,
             color=INK2, va="top")
    fig.text(0.015, 0.15 / fig_h, note_text, fontsize=8.4, color=MUTED, va="bottom", linespacing=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfa")
    plt.close(fig)
