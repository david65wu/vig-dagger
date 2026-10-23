"""The experiment's results as files, without W&B: ``results/REPORT.md`` (every table) and ``results/figures/*.png``.

Built from what the pipeline has written so far, so it can be regenerated at any time (``vig-dagger report``; the
``analyze_<round>`` and ``summarize`` jobs also refresh it). Sections whose inputs don't exist yet say so.

  REPORT.md                              progress, evaluation, drivers by resolution, where evaluations fail,
                                         DAgger collections, per-stage expert share, trainings, figure list
  figures/eval_success.png               success per training set: mean over seeds (bar) and each seed (dots)
  figures/eval_drivers_by_level.png      success of the MoE and of the per-resolution policies at each level
  figures/dagger_<round>_distributions.png   per-episode expert share, takeovers, length, first takeover and
                                         takeover lengths, VIG and flow loss overlaid
  figures/dagger_<round>_scores.png      score on policy- and expert-driven steps, with the thresholds
  figures/dagger_<round>_stages.png      expert share per task stage
  figures/dagger_<round>_<scorer>_timelines.png   six episodes' score per step (written by the analysis)
  figures/training_<set>.png             training and validation loss of each seed (drivers in training_drivers.png)
  training_summary.csv                   per training: updates, last training loss, final and best validation loss
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from vig_dagger import stages
from vig_dagger.config import Paths, training_sets


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def _num(x, digits: int = 4) -> str:
    return "–" if x is None else f"{x:.{digits}g}"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- trainings


def training_runs(cfg: dict) -> list[tuple[str, str, Path]]:
    """(group, label, run directory) of every training that has started."""
    paths, runs = Paths.of(cfg), []
    for name in training_sets(cfg):
        for s in cfg["retrain"]["seeds"]:
            run = paths.retrain_run(name, int(s))
            if (run / "metrics.jsonl").exists():
                runs.append((name, f"retrain/{name}/seed{s}", run))
    for d in sorted((paths.driver_dir("per_resolution")).glob("res*"), key=lambda p: int(p.name[3:])):
        if (d / "metrics.jsonl").exists():
            runs.append(("drivers", f"drivers/per_resolution/{d.name}", d))
    if (paths.driver_dir("moe") / "metrics.jsonl").exists():
        runs.append(("drivers", "drivers/moe", paths.driver_dir("moe")))
    return runs


def _curves(run: Path) -> tuple[list, list, list, list]:
    train_s, train_l, val_s, val_l = [], [], [], []
    for line in (run / "metrics.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "train" and "train/loss" in e:
            train_s.append(e["train/step"]); train_l.append(e["train/loss"])
        elif e.get("event") == "validation" and "val/loss" in e:
            val_s.append(e["train/step"]); val_l.append(e["val/loss"])
    return train_s, train_l, val_s, val_l


def training_summary(cfg: dict) -> list[dict]:
    rows = []
    for group, label, run in training_runs(cfg):
        ts, tl, vs, vl = _curves(run)
        try:
            info = json.loads((run / "run_config.json").read_text())
            total, demos = info["training"]["max_updates"], info.get("_info", {}).get("training_demos")
        except (OSError, KeyError, ValueError):
            total, demos = None, None
        best = int(np.argmin(vl)) if vl else None
        finished = (run / "latest.pt").exists()
        updates = total if finished and total else (ts[-1] if ts else 0)  # losses are logged every 500 updates
        rows.append(dict(run=label, training_demos=demos, updates=updates, max_updates=total, finished=finished, last_train_loss=float(np.mean(tl[-5:])) if tl else None,
                         final_val_loss=vl[-1] if vl else None, best_val_loss=vl[best] if vl else None,
                         best_val_step=vs[best] if vl else None))
    return rows


def figures_training(cfg: dict, figures: Path) -> list[str]:
    plt, made = _plt(), []
    groups: dict[str, list] = {}
    for group, label, run in training_runs(cfg):
        groups.setdefault(group, []).append((label, run))
    for group, runs in groups.items():
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for label, run in runs:
            ts, tl, vs, vl = _curves(run)
            name = label.split("/")[-1]
            ax[0].plot(ts, tl, lw=1, label=name)
            ax[1].plot(vs, vl, lw=1, marker=".", label=name)
        for a, title in zip(ax, ("training loss", "validation loss")):
            a.set_title(title, fontsize=10); a.set_xlabel("update"); a.set_yscale("log"); a.legend(fontsize=7)
        fig.suptitle(f"training: {group}", fontsize=10)
        fig.tight_layout()
        name = f"training_{group}.png"
        fig.savefig(figures / name, dpi=90); plt.close(fig)
        made.append(name)
    return made


# ---------------------------------------------------------------- DAgger


def dagger_results(cfg: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for r in cfg["rounds"]:
        for s in cfg["dagger"]["scorers"]:
            path = Paths.of(cfg).results() / f"dagger_{r}_{s}.json"
            if path.exists():
                out[(r, s)] = json.loads(path.read_text())
    return out


def figures_dagger(cfg: dict, figures: Path) -> list[str]:
    plt, made, results = _plt(), [], Paths.of(cfg).results()
    colors = {"vig": "#3b6ea5", "flow": "#d1793b"}
    for r in cfg["rounds"]:
        eps = {s: _read_csv(results / f"dagger_{r}_{s}_episodes.csv") for s in cfg["dagger"]["scorers"]}
        tos = {s: _read_csv(results / f"dagger_{r}_{s}_takeovers.csv") for s in cfg["dagger"]["scorers"]}
        if not any(eps.values()):
            continue
        keys = [("expert_share", "expert share per episode"), ("takeovers", "takeovers per episode"),
                ("steps", "episode length (steps)"), ("first_take_step", "step of the first takeover")]
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
        for ax, (key, title) in zip(axes, keys):
            for s, rows in eps.items():
                values = [float(x[key]) for x in rows if x.get(key) not in (None, "")]
                if values:
                    ax.hist(values, bins=30, alpha=0.5, color=colors.get(s), label=f"{s} (n={len(values)})")
            ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
        for s, rows in tos.items():
            values = [float(x["length"]) for x in rows]
            if values:
                axes[4].hist(values, bins=30, alpha=0.5, color=colors.get(s), label=s)
        axes[4].set_title("takeover length (expert steps)", fontsize=9); axes[4].legend(fontsize=7)
        fig.suptitle(f"round {r}: DAgger attempts", fontsize=10); fig.tight_layout()
        fig.savefig(figures / f"dagger_{r}_distributions.png", dpi=90); plt.close(fig)
        made.append(f"dagger_{r}_distributions.png")

        fig, axes = plt.subplots(1, len(cfg["dagger"]["scorers"]), figsize=(6 * len(cfg["dagger"]["scorers"]), 3.2), squeeze=False)
        for ax, s in zip(axes[0], cfg["dagger"]["scorers"]):
            path = results / f"dagger_{r}_{s}_scores.npz"
            res = results / f"dagger_{r}_{s}.json"
            if not path.exists() or not res.exists():
                continue
            scores, settings = np.load(path), json.loads(res.read_text())["settings"]
            hi = np.percentile(np.concatenate([scores["policy"], scores["expert"]]), 99.5) if len(scores["policy"]) else 1
            for who, color in (("policy", "#3b6ea5"), ("expert", "#e8a33d")):
                if len(scores[who]):
                    ax.hist(np.clip(scores[who], 0, hi), bins=60, alpha=0.55, color=color, density=True, label=f"{who}-driven steps")
            ax.axvline(settings["take_threshold"], color="#c0392b", ls="--", lw=1, label=f"take {settings['take_spec']}")
            ax.axvline(settings["return_threshold"], color="#27ae60", ls="--", lw=1, label=f"return {settings['return_spec']}")
            ax.set_title(f"{s}: score per step", fontsize=9); ax.legend(fontsize=7)
        fig.suptitle(f"round {r}: scores", fontsize=10); fig.tight_layout()
        fig.savefig(figures / f"dagger_{r}_scores.png", dpi=90); plt.close(fig)
        made.append(f"dagger_{r}_scores.png")

        fig, ax = plt.subplots(figsize=(8, 3.2))
        width = 0.8 / max(1, len(cfg["dagger"]["scorers"]))
        x = np.arange(len(stages.STAGES))
        for i, s in enumerate(cfg["dagger"]["scorers"]):
            res = results / f"dagger_{r}_{s}.json"
            if not res.exists():
                continue
            st = json.loads(res.read_text())["metrics"]["all"].get("stages", {})
            ax.bar(x + i * width, [st.get(n, {}).get("expert_share") or 0 for n in stages.STAGES], width,
                   color=colors.get(s), label=s)
        ax.set_xticks(x + width / 2, stages.STAGES, fontsize=8); ax.set_ylabel("expert share"); ax.legend(fontsize=8)
        ax.set_title(f"round {r}: share of steps the expert drove, per task stage", fontsize=10); fig.tight_layout()
        fig.savefig(figures / f"dagger_{r}_stages.png", dpi=90); plt.close(fig)
        made.append(f"dagger_{r}_stages.png")
    for r in cfg["rounds"]:
        for s in cfg["dagger"]["scorers"]:
            if (figures / f"dagger_{r}_{s}_timelines.png").exists():
                made.append(f"dagger_{r}_{s}_timelines.png")
    return made


# ---------------------------------------------------------------- evaluation


def figures_eval(summary: dict, figures: Path) -> list[str]:
    plt, made = _plt(), []
    sets = [e for e in summary.get("datasets", {}).values() if e["factor"] == 1]
    if sets:
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(sets)), 3.6))
        for i, e in enumerate(sets):
            ax.bar(i, e["success_mean"], color="#9bb7d4", yerr=e["success_std"], capsize=3)
            ax.scatter([i] * len(e["success_by_seed"]), list(e["success_by_seed"].values()), color="#22313f", s=12, zorder=3)
        ax.set_xticks(range(len(sets)), [e["dataset"] for e in sets], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("success rate"); ax.set_ylim(0, 1)
        ax.set_title("success per training set (bar: mean ± std over seeds; dots: seeds)", fontsize=10)
        fig.tight_layout(); fig.savefig(figures / "eval_success.png", dpi=90); plt.close(fig)
        made.append("eval_success.png")
    levels = summary.get("drivers_by_level", {})
    if levels:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        xs = sorted(levels, key=int)
        for key, color, label in (("moe", "#3b6ea5", "MoE at that level"), ("per_resolution", "#d1793b", "policy trained at that level")):
            pts = [(i, levels[f][key]) for i, f in enumerate(xs) if key in levels[f]]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=color, label=label)
        ax.set_xticks(range(len(xs)), ["full" if f == "1" else f"1/{f}" for f in xs])
        ax.set_ylabel("success rate"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
        ax.set_title("driving policies by image resolution", fontsize=10)
        fig.tight_layout(); fig.savefig(figures / "eval_drivers_by_level.png", dpi=90); plt.close(fig)
        made.append("eval_drivers_by_level.png")
    return made


# ---------------------------------------------------------------- report


def write(cfg: dict) -> Path:
    paths = Paths.of(cfg)
    results, figures = paths.results(), paths.results() / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    lines = [f"# {cfg['name']}: results", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{paths.root}`. Regenerate with `vig-dagger report`.", ""]

    state = paths.logs() / "queue_state.json"
    if state.exists():
        jobs = json.loads(state.read_text())["jobs"]
        counts: dict[str, int] = {}
        for entry in jobs.values():
            counts[entry["state"]] = counts.get(entry["state"], 0) + 1
        lines += ["**Progress:** " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: -kv[1]))
                  + f" (of {len(jobs)} jobs)." + ("" if counts.get("done") == len(jobs) else " Partial results."), ""]

    # evaluation
    summary_path = results / "eval_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    lines += ["## Evaluation", ""]
    if summary.get("datasets"):
        n_eval = cfg["evaluation"]["num_episodes"]
        lines += [f"Success rate of the retrained policies (dense, full resolution), {n_eval} scenes per seed, the same "
                  "scenes for every policy. *Only this / only base*: scenes solved by this policy and not by `base` of the "
                  "same seed, and the reverse, summed over seeds; *p*: two-sided sign test on those.", "",
                  "| training set | " + " | ".join(f"seed {s}" for s in cfg["retrain"]["seeds"]) + " | mean ± std | only this / only base | p |",
                  "|---|" + "---|" * len(cfg["retrain"]["seeds"]) + "---|---|---|"]
        for e in summary["datasets"].values():
            if e["factor"] != 1:
                continue
            seeds = [_pct(e["success_by_seed"].get(str(s), e["success_by_seed"].get(s))) for s in cfg["retrain"]["seeds"]]
            vs = f"{e['vs_base_only_this']} / {e['vs_base_only_base']}" if "vs_base_only_this" in e else "–"
            p = _num(e.get("vs_base_sign_test_p"), 3)
            lines.append(f"| `{e['dataset']}` | " + " | ".join(seeds) + f" | {_pct(e['success_mean'])} ± {_pct(e['success_std'])} | {vs} | {p} |")
        lines.append("")
    else:
        lines += ["No evaluation summary yet (`results/eval_summary.json` is written by the `summarize` job).", ""]

    levels = summary.get("drivers_by_level", {})
    if levels:
        lines += ["### Driving policies by image resolution", "",
                  "The MoE (round `moe`) at each level, and round `per_resolution`'s policy trained at that level "
                  f"(`retrain/base/seed{cfg['retrain']['seeds'][0]}` at full resolution).", "",
                  "| level | MoE | per-resolution policy |", "|---|---|---|"]
        for f in sorted(levels, key=int):
            lines.append(f"| {'full' if f == '1' else '1/' + f} | {_pct(levels[f].get('moe'))} | {_pct(levels[f].get('per_resolution'))} |")
        lines.append("")

    pols = summary.get("policies", {})
    if pols:
        lines += ["### Where evaluation episodes end", "",
                  "Furthest stage reached by the failed episodes, and failure flags over all episodes, per policy and level.", "",
                  "| policy | level | success | " + " | ".join(stages.STAGES[:-1]) + " | " + " | ".join(stages.FLAGS) + " |",
                  "|---|---|---|" + "---|" * (len(stages.STAGES) - 1 + len(stages.FLAGS))]
        for v in pols.values():
            failed = max(1, sum(v["failed_at"].values()))
            row = [f"`{v['label']}`", "full" if v["factor"] == 1 else f"1/{v['factor']}", _pct(v["success_rate"])]
            row += [_pct(v["failed_at"].get(s, 0) / failed) if sum(v["failed_at"].values()) else "–" for s in stages.STAGES[:-1]]
            row += [_pct(v["flags"].get(k)) for k in stages.FLAGS]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # DAgger
    lines += ["## DAgger collections", ""]
    dagger = dagger_results(cfg)
    if dagger:
        lines += ["Per round and scorer, over every attempt. Distributions over episodes as median [p25, p75].", "",
                  "| round | scorer | take / return threshold | attempts | successes | success rate | expert share (pooled) "
                  "| expert share per episode | takeovers per episode | returns per episode | no takeover | takeover length | failure reasons |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (r, s), d in dagger.items():
            m, st = d["metrics"], d["settings"]
            a = m["all"]
            pe = a.get("per_episode", {})

            def q(key, digits=2, pct=False):
                x = pe.get(key)
                if not x:
                    return "–"
                f = (lambda v: _pct(v)) if pct else (lambda v: f"{v:.{digits}g}")
                return f"{f(x['median'])} [{f(x['p25'])}, {f(x['p75'])}]"

            tl = a.get("per_takeover", {}).get("length")
            tl_text = f"{tl['median']:.3g} [{tl['p25']:.3g}, {tl['p75']:.3g}]" if tl else "–"
            reasons = ", ".join(f"{k} {v}" for k, v in m.get("failed", {}).get("reasons", {}).items()) or "–"
            thresholds = (f"{st['take_threshold']:.4g} ({st['take_spec']}, ×{st['take_count']}) / "
                          f"{st['return_threshold']:.4g} ({st['return_spec']}, ×{st['return_count']})")
            cells = [r, s, thresholds, m["collection"]["attempts"], m["collection"]["successes"],
                     _pct(m["collection"]["success_rate"]), _pct(a.get("pooled", {}).get("expert_share")),
                     q("expert_share", pct=True), q("takeovers"), q("returns"), _pct(a.get("no_takeover_share")),
                     tl_text, reasons]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        lines += ["", "Successful vs failed attempts:", "",
                  "| round | scorer | group | episodes | expert share (pooled) | expert share per episode, median | takeovers per episode, median | steps, median |",
                  "|---|---|---|---|---|---|---|---|"]
        for (r, s), d in dagger.items():
            for group in ("success", "failed"):
                g = d["metrics"][group]
                if not g.get("episodes"):
                    continue
                pe = g["per_episode"]
                lines.append(f"| {r} | {s} | {group} | {g['episodes']} | {_pct(g['pooled']['expert_share'])} | "
                             f"{_pct(pe['expert_share']['median'])} | {pe['takeovers']['median']:.3g} | {pe['steps']['median']:.3g} |")
        lines += ["", "Expert share per task stage (all attempts):", "",
                  "| round | scorer | " + " | ".join(stages.STAGES) + " |", "|---|---|" + "---|" * len(stages.STAGES)]
        for (r, s), d in dagger.items():
            st = d["metrics"]["all"].get("stages", {})
            lines.append(f"| {r} | {s} | " + " | ".join(_pct(st.get(n, {}).get("expert_share")) for n in stages.STAGES) + " |")
        lines.append("")
        lines += ["Per-episode and per-takeover data: `results/dagger_<round>_<scorer>_episodes.csv`, `_takeovers.csv`; "
                  "every metric: `results/dagger_<round>_<scorer>.json`.", ""]
    else:
        lines += ["No DAgger analysis yet (written by the `analyze_<round>` jobs when a round's collections finish).", ""]

    # trainings
    rows = training_summary(cfg)
    if rows:
        with open(results / "training_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        lines += ["## Trainings", "", "| run | training demos | updates | finished | last training loss | final / best validation loss |",
                  "|---|---|---|---|---|---|"]
        for row in rows:
            total = f"{row['max_updates']:,}" if row["max_updates"] else "?"
            lines.append(f"| `{row['run']}` | {row['training_demos'] or '–'} | {row['updates']:,} / {total} | "
                         f"{'yes' if row['finished'] else 'no'} | {_num(row['last_train_loss'])} | "
                         f"{_num(row['final_val_loss'])} / {_num(row['best_val_loss'])} |")
        lines.append("")

    # figures
    made = figures_eval(summary, figures) + figures_dagger(cfg, figures) + figures_training(cfg, figures)
    if made:
        lines += ["## Figures", ""] + [f"- [{name}](figures/{name})" for name in made] + [""]
    path = results / "REPORT.md"
    path.write_text("\n".join(lines))
    print(f"[report] {path} ({len(made)} figures)", flush=True)
    return path
