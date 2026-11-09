"""The experiment's results as files, without W&B: ``results/REPORT.md`` and ``results/figures/*.png``.

Built from what the pipeline has written so far, so it can be regenerated at any time (``vig-dagger report``; every
collection's ``analyze_*`` job and the ``summarize`` job also refresh it). Sections whose inputs don't exist yet say so.

  REPORT.md                     progress; the three numbers per setting (mean ± std over the MoE seeds); every
                                collection; the thresholds of each MoE; the trainings
  figures/collections.png       the three numbers per setting (summary.py)
  figures/moves/spheres_{held,progress}_sd<s>_p<q>.png   the random moves of MoE s at threshold q, VIG and flow loss
                                side by side, seen from the third-person camera (move_figures.py)
  figures/eval_success.png      success of the retrained policies per setting, with the MoEs' (evaluation.py)
  figures/training_moe.png      training and validation loss of the MoEs trained here (training_retrain.png: the
                                retrained policies)
  summary.json, summary.csv     the numbers behind the tables (summary.py)
  training_summary.csv          per training: updates, last training loss, final and best validation loss
"""

from __future__ import annotations

import csv
import json
import time

import numpy as np

from vig_dagger import move_figures
from vig_dagger import summary as summary_module
from vig_dagger.config import Paths, collections, downloaded


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _pct(x, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f} %"


def _num(x, digits: int = 4) -> str:
    return "–" if x is None else f"{x:.{digits}g}"


def _mean_std(cell: dict) -> str:
    if cell["mean"] is None:
        return "–"
    return _pct(cell["mean"]) + ("" if cell["std"] is None else f" ± {100 * cell['std']:.1f}")


# ---------------------------------------------------------------- trainings


def training_runs(cfg: dict) -> list[tuple[str, object]]:
    """(label, run directory) of every training whose log is here: the MoEs, then the retrained policies."""
    paths = Paths.of(cfg)
    runs = [(f"moe/sd{s}", paths.moe_dir(int(s))) for s in cfg["moe"]["seeds"]]
    runs += [(f"retrain/{c.id}/seed{int(s)}", paths.retrain_run(c, int(s))) for c in collections(cfg) for s in cfg["retrain"]["seeds"]]
    return [(label, run) for label, run in runs if (run / "metrics.jsonl").exists()]


def _curves(run) -> tuple[list, list, list, list]:
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
    for label, run in training_runs(cfg):
        ts, tl, vs, vl = _curves(run)
        try:
            config_file = run / "run_config.json" if (run / "run_config.json").exists() else run / "train_config.json"
            info = json.loads(config_file.read_text())
            total, demos = info["training"]["max_updates"], info.get("_info", {}).get("training_demos")
        except (OSError, KeyError, ValueError):
            total, demos = None, None
        best = int(np.argmin(vl)) if vl else None
        finished = (run / "latest.pt").exists()
        updates = total if finished and total else (ts[-1] if ts else 0)  # losses are logged every 500 updates
        rows.append(dict(run=label, training_demos=demos, updates=updates, max_updates=total, finished=finished,
                         last_train_loss=float(np.mean(tl[-5:])) if tl else None, final_val_loss=vl[-1] if vl else None,
                         best_val_loss=vl[best] if vl else None, best_val_step=vs[best] if vl else None))
    return rows


def figure_training(cfg: dict, figures) -> list[str]:
    made = []
    groups = {"moe": [r for r in training_runs(cfg) if r[0].startswith("moe/")],
              "retrain": [r for r in training_runs(cfg) if r[0].startswith("retrain/")]}
    for group, runs in groups.items():
        if not runs:
            continue
        plt = _plt()
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for label, run in runs:
            ts, tl, vs, vl = _curves(run)
            name = label.split("/")[1] if group == "retrain" else label
            ax[0].plot(ts, tl, lw=1, label=name)
            ax[1].plot(vs, vl, lw=1, marker=".", label=name)
        for a, title in zip(ax, ("training loss", "validation loss")):
            a.set_title(title, fontsize=10); a.set_xlabel("update"); a.set_yscale("log")
            if len(runs) <= 12:
                a.legend(fontsize=6)
        fig.suptitle("MoE trainings" if group == "moe" else "retrained policies (dense, full resolution)", fontsize=10)
        fig.tight_layout()
        fig.savefig(figures / f"training_{group}.png", dpi=90); plt.close(fig)
        made.append(f"training_{group}.png")
    return made


# ---------------------------------------------------------------- report


def write(cfg: dict):
    paths = Paths.of(cfg)
    results, figures = paths.results(), paths.results() / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = summary_module.summarize(cfg)
    from vig_dagger import evaluation

    evals = evaluation.summarize(cfg) if (paths.root / "eval").is_dir() else None
    d, n_demos = cfg["dagger"], int(cfg["dagger"]["num_demos"])
    lines = [f"# {cfg['name']}: results", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{paths.root}`. Regenerate with `vig-dagger report`.", ""]

    state = paths.logs() / "queue_state.json"
    from vig_dagger.config import board_mode

    if board_mode(cfg):  # several machines: the tasks on the board, not this machine's queue
        try:
            from vig_dagger import board

            b = board.get(cfg)
            b.refresh(force=True)
            total = len(cfg["moe"]["seeds"]) + len(collections(cfg)) + 1
            done = sum(v["done"] for v in b.state.values())
            if not b.is_done("summary") and done == total - 1:  # this is the summary task writing its report
                done = total
            lines += [f"**Progress:** {done} of {total} tasks done on the task board `{b.spec}`."
                      + ("" if done == total else " Partial results."), ""]
        except Exception as exc:  # the board is unreachable: say so, the rest of the report is local
            lines += [f"**Progress:** the task board could not be read ({type(exc).__name__}).", ""]
    elif state.exists():
        jobs = json.loads(state.read_text())["jobs"]
        counts: dict[str, int] = {}
        for name, entry in jobs.items():
            # the summary job writes this report while it runs: count it as done
            st = "done" if name == "summarize" and entry["state"] == "running" else entry["state"]
            counts[st] = counts.get(st, 0) + 1
        lines += ["**Progress:** " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: -kv[1]))
                  + f" (of {len(jobs)} jobs on this machine)." + ("" if counts.get("done") == len(jobs) else " Partial results."), ""]
    lines += [f"**Collections finished and analyzed:** {summary['collections_finished']} of {summary['collections_total']}"
              + (f" (and {summary['collections_analyzed'] - summary['collections_finished']} analyzed mid-run, not in the "
                 "means)" if summary["collections_analyzed"] > summary["collections_finished"] else "") + ".", ""]

    # the three numbers per setting
    lines += ["## DAgger collections", "",
              f"One collection per MoE seed ({', '.join(str(s) for s in cfg['moe']['seeds'])}), scorer, threshold and "
              f"variant, each with {n_demos} successful episode{'s' if n_demos != 1 else ''}. The threshold is a percentile of the MoE's own score "
              f"on demo frames; the expert takes over after {d['take_count']} scores above it and the policy takes back "
              f"after {d['return_count']} scores below it. *Random move*: at every takeover the gripper first moves to a "
              f"random pose within {d['random_move']['radius'] * 100:g} cm and {d['random_move']['rot_deg']:g}° "
              "(unscored, gripper held), then the expert replans.", "",
              "![the three numbers per setting](figures/collections.png)", ""]
    settings = [s for s in summary["settings"] if s["seeds"]]
    if settings:
        lines += ["Per setting, mean ± std over the MoE seeds (percentage points). *Expert share* is pooled over the steps "
                  "of the collection's DAgger data (its first successful episodes) or of its failed attempts; the random "
                  "move's share is separate (the policy drove the rest); *median per episode*: the mean over seeds of the "
                  "per-episode median.", "",
                  "| scorer | variant | threshold | MoE seeds | success rate | successes / attempts | expert share of the data "
                  "| random move | median per episode | expert share of the failed attempts | random move |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in settings:
            move = s["variant"] == "random_move"
            lines.append(
                f"| {s['scorer']} | {s['variant'].replace('_', ' ')} | {s['threshold']} | {', '.join(str(x) for x in s['seeds'])} "
                f"| {_mean_std(s['success_rate'])} | {s['successes']} / {s['attempts']} | {_mean_std(s['data_expert'])} "
                f"| {_mean_std(s['data_move']) if move else '–'} | {_pct(s['data_expert_median']['mean'])} "
                f"| {_mean_std(s['failed_expert'])} | {_mean_std(s['failed_move']) if move else '–'} |")
        lines.append("")
    rows = summary["collections"]
    if rows:
        lines += ["### Every collection", "",
                  "Expert share pooled over steps / median per episode; random-move shares separate. Takeovers per attempt: "
                  "median over all attempts.", "",
                  "| collection | threshold | attempts | successes | success rate | data: expert | data: random move "
                  "| failed: expert | failed: random move | takeovers | random moves (skipped) | failure reasons |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            move = r["variant"] == "random_move"
            reasons = ", ".join(f"{k} {v}" for k, v in r["reasons"].items()) or "–"
            lines.append(
                f"| `{r['collection']}`{'' if r['finished'] else ' (running)'} | {r['threshold']} = {_num(r['threshold_value'])} "
                f"| {r['attempts']} | {r['successes']} "
                f"| {_pct(r['success_rate'])} | {_pct(r['data_expert'])} / {_pct(r['data_expert_median'])} "
                f"| {(_pct(r['data_move']) + ' / ' + _pct(r['data_move_median'])) if move else '–'} "
                f"| {_pct(r['failed_expert'])} / {_pct(r['failed_expert_median'])} "
                f"| {(_pct(r['failed_move']) + ' / ' + _pct(r['failed_move_median'])) if move else '–'} "
                f"| {_num(r['takeovers_median'], 3)} | {(str(r['random_moves']) + ' (' + str(r['moves_skipped']) + ')') if move else '–'} "
                f"| {reasons} |")
        lines += ["", "Per collection: every metric in `results/collections/<collection>.json`, one row per attempt in "
                  "`_episodes.csv`, per takeover in `_takeovers.csv`, the data of a page of the collection in "
                  "`_page.json.gz`.", ""]
    else:
        lines += ["No collection analyzed yet (each collection's `analyze_*` job writes its metrics when it finishes).", ""]

    # random moves
    moves = move_figures.figures(cfg, figures)
    if moves:
        lines += ["### Random moves", "",
                  "Every random move of the random-move collections' DAgger data, seen from the third-person camera: the "
                  f"{d['random_move']['radius'] * 100:g} cm sphere the target was drawn from (centred on the gripper tip at "
                  "the take-over), a dot at its centre, an arrow to where the tip ended; VIG and flow loss side by side. "
                  "*Held*: colour = plug held at the take-over; *progress*: colour = take-over step / episode length.", "",
                  "| MoE seed | " + " | ".join(d["thresholds"]) + " |", "|---|" + "---|" * len(d["thresholds"])]
        by = {(seed, spec): names for seed, spec, names in moves}
        for seed in (int(x) for x in cfg["moe"]["seeds"]):
            cells = []
            for spec in d["thresholds"]:
                names = by.get((seed, spec), [])
                cells.append(" · ".join(f"[{n.split('_')[1]}](figures/{n})" for n in names) or "–")
            lines.append(f"| {seed} | " + " | ".join(cells) + " |")
        first = moves[0]
        lines += ["", f"MoE seed {first[0]} at {first[1]}:", ""] + [f"![{n}](figures/{n})" for n in first[2]] + [""]

    # retrained policies and evaluations
    lines += ["## Retrained policies", ""]
    if evals and (evals["retrained"] or evals["moes"]):
        ev = cfg["evaluation"]
        lines += [f"Per collection a dense full-resolution policy (no experts) trained on the {cfg['demos']['num_demos']} base "
                  f"demos + the collection's {d['num_demos']} DAgger episodes ({cfg['training']['epochs']} epochs, training seed "
                  f"{', '.join(str(x) for x in cfg['retrain']['seeds'])}), evaluated on {ev['num_episodes']} scenes "
                  f"({ev['seed_start']}–{ev['seed_start'] + ev['num_episodes'] - 1}, the same for every policy). "
                  "*Only this / only MoE*: scenes the retrained policy solved and the MoE that drove its collection did not, "
                  "and the reverse, summed over the MoE seeds; *p*: two-sided sign test on those.", "",
                  "![success of the retrained policies](figures/eval_success.png)", ""]
        if evals["moes"]:
            lines += ["The driving MoEs at full resolution: " + ", ".join(f"seed {k} {_pct(v)}" for k, v in sorted(evals["moes"].items()))
                      + ".", ""]
        rows = [x for x in evals["settings"] if x["moe_seeds"]]
        if rows:
            lines += ["| scorer | variant | threshold | MoE seeds | success, mean ± std | by MoE seed | only this / only MoE | p |",
                      "|---|---|---|---|---|---|---|---|"]
            for x in rows:
                cell = dict(mean=x["success_mean"], std=x["success_std"])
                by = ", ".join(f"{k}: {_pct(v)}" for k, v in x["success_by_moe_seed"].items())
                lines.append(f"| {x['scorer']} | {x['variant'].replace('_', ' ')} | {x['threshold']} | "
                             f"{', '.join(str(k) for k in x['moe_seeds'])} | {_mean_std(cell)} | {by} | "
                             f"{x['only_this']} / {x['only_moe']} | {_num(x['sign_test_p'], 3)} |")
            lines.append("")
        if evals["retrained"]:
            lines += ["Every retrained policy:", "", "| policy | success | its MoE | only this / only MoE | p | successful episodes, median steps |",
                      "|---|---|---|---|---|---|"]
            for r in evals["retrained"]:
                pol = evals["policies"].get(r["label"], {})
                lines.append(f"| `{r['label']}` | {_pct(r['success_rate'])} | {_pct(r.get('moe_success_rate'))} | "
                             f"{r.get('only_this', '–')} / {r.get('only_moe', '–')} | {_num(r.get('sign_test_p'), 3)} | "
                             f"{_num(pol.get('success_steps_median'), 4)} |")
            lines.append("")
        running = [k for k, v in evals["policies"].items() if not v["complete"]]
        if running:
            lines += [f"Still being evaluated (not in the tables): {', '.join(f'`{k}`' for k in running)}.", ""]
        lines += ["Every episode: `results/eval_episodes.csv`; every number: `results/eval_summary.json`.", ""]
    else:
        lines += ["No evaluation yet (each retraining task trains on its collection's DAgger data, then evaluates).", ""]

    # thresholds
    lines += ["## Thresholds", "", "Percentiles of each MoE's score on demo frames (every "
              f"{cfg['thresholds']['frame_stride']}th frame of the first {cfg['thresholds']['num_demos']} demos).", "",
              "| MoE seed | source | frames | " + " | ".join(f"VIG {p}" for p in d["thresholds"]) + " | "
              + " | ".join(f"flow {p}" for p in d["thresholds"]) + " |",
              "|---|---|---|" + "---|" * (2 * len(d["thresholds"]))]
    for seed in (int(s) for s in cfg["moe"]["seeds"]):
        path = paths.thresholds(seed)
        source = "downloaded" if downloaded(cfg, seed) else "trained here"
        if path.exists():
            t = json.loads(path.read_text())
            lines.append(f"| {seed} | {source} | {t['frames']} | " + " | ".join(_num(t["vig"].get(p)) for p in d["thresholds"])
                         + " | " + " | ".join(_num(t["flow"].get(p)) for p in d["thresholds"]) + " |")
        else:
            lines.append(f"| {seed} | {source} | not yet | " + " | ".join("–" for _ in range(2 * len(d["thresholds"]))) + " |")
    lines.append("")

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

    made = (["collections.png"] if (figures / "collections.png").exists() else []) \
        + (["eval_success.png"] if (figures / "eval_success.png").exists() else []) + figure_training(cfg, figures) \
        + [n for _, _, names in moves for n in names]
    if made:
        lines += ["## Figures", ""] + [f"- [{name}](figures/{name})" for name in made] + [""]
    path = results / "REPORT.md"
    path.write_text("\n".join(lines))
    print(f"[report] {path} ({len(made)} figures)", flush=True)
    return path

