"""Evaluation summary (part 4): success per policy and level, aggregated over training seeds, paired comparisons.

Per evaluated policy and level: success rate, steps of successful episodes, where episodes end (furthest stage,
the stage a failure ended in) and how often each failure flag was raised.

Per training set (``base``, ``base_plus_planner``, ``base_plus_vig``, ``base_plus_flow``) and level: the success
rate of each seed, their mean and standard deviation, and a paired comparison with ``base``: over all seeds,
scenes solved by one and not the other (seed i against seed i), with a two-sided sign test on those.

Outputs: ``results/eval_summary.json``, ``results/eval_episodes.csv``, and W&B runs ``<name>/eval/<policy>`` (one
per evaluated policy: per-level scalars and an episode table) and ``<name>/eval/summary`` (the comparison table and
success bar charts).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vig_dagger import stages


def load(eval_dir: Path) -> dict[tuple[str, int], list[dict]]:
    """{(policy label, factor): episode records} from ``eval/<label>/level<f>/*.jsonl``."""
    out: dict[tuple[str, int], list[dict]] = {}
    for path in sorted(eval_dir.glob("**/level*/*.jsonl")):
        label = str(path.parent.parent.relative_to(eval_dir))
        factor = int(path.parent.name[5:])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out.setdefault((label, factor), []).extend(rows)
    for key in out:
        out[key] = sorted({r["seed"]: r for r in out[key]}.values(), key=lambda r: r["seed"])
    return out


def policy_metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    failed = [r for r in rows if not r["success"]]
    return dict(
        episodes=n, success_rate=len(succ) / n if n else None,
        success_steps_mean=float(np.mean([r["steps"] for r in succ])) if succ else None,
        success_steps_median=float(np.median([r["steps"] for r in succ])) if succ else None,
        furthest_stage={s: sum(r["furthest_stage"] == s for r in rows) for s in stages.STAGES},
        failed_at={s: sum(r["failed_at"] == s for r in failed) for s in stages.STAGES},
        flags={k: float(np.mean([r[f"flag_{k}"] for r in rows])) if rows else None for k in stages.FLAGS},
        failed_flags={k: float(np.mean([r[f"flag_{k}"] for r in failed])) if failed else None for k in stages.FLAGS})


def sign_test(wins: int, losses: int) -> float | None:
    if wins + losses == 0:
        return None
    from scipy.stats import binomtest

    return float(binomtest(wins, wins + losses, 0.5).pvalue)


def summarize(cfg: dict, eval_dir: Path, results: Path) -> dict[str, Any]:
    from vig_dagger import tracking

    data = load(eval_dir)
    per_policy = {f"{label}@{factor}": dict(label=label, factor=factor, **policy_metrics(rows))
                  for (label, factor), rows in sorted(data.items())}
    # retrain/<dataset>/seed<s> labels grouped by dataset
    groups: dict[tuple[str, int], dict[int, list[dict]]] = {}
    for (label, factor), rows in data.items():
        parts = label.split("/")
        if len(parts) == 3 and parts[0] == "retrain":
            groups.setdefault((parts[1], factor), {})[int(parts[2][4:])] = rows
    by_dataset = {}
    for (name, factor), seeds in sorted(groups.items()):
        rates = {s: float(np.mean([r["success"] for r in rows])) for s, rows in sorted(seeds.items())}
        entry = dict(dataset=name, factor=factor, seeds=sorted(seeds), success_by_seed=rates,
                     success_mean=float(np.mean(list(rates.values()))), success_std=float(np.std(list(rates.values()))),
                     episodes=sum(len(rows) for rows in seeds.values()))
        base = groups.get(("base", factor))
        if base is not None and name != "base":
            wins = losses = 0
            for s, rows in seeds.items():
                if s not in base:
                    continue
                ref = {r["seed"]: r["success"] for r in base[s]}
                for r in rows:
                    if r["seed"] in ref:
                        wins += int(r["success"] and not ref[r["seed"]])
                        losses += int(ref[r["seed"]] and not r["success"])
            entry.update(vs_base_only_this=wins, vs_base_only_base=losses, vs_base_sign_test_p=sign_test(wins, losses))
        by_dataset[f"{name}@{factor}"] = entry
    summary = dict(policies=per_policy, datasets=by_dataset)
    results.mkdir(parents=True, exist_ok=True)
    (results / "eval_summary.json").write_text(json.dumps(summary, indent=1))
    rows_out = [dict(policy=label, **{k: v for k, v in r.items() if not isinstance(v, dict)})
                for (label, _), rows in sorted(data.items()) for r in rows]
    if rows_out:
        with open(results / "eval_episodes.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_out)
    for key, entry in by_dataset.items():
        print(f"[eval] {key}: {entry['success_mean']:.3f} +- {entry['success_std']:.3f} over seeds {entry['seeds']}"
              + (f", vs base {entry.get('vs_base_only_this')}:{entry.get('vs_base_only_base')} p={entry.get('vs_base_sign_test_p')}"
                 if "vs_base_only_this" in entry else ""), flush=True)

    if cfg["wandb"].get("enabled", True):
        import wandb

        for label in sorted({label for label, _ in data}):
            run = tracking.start(cfg, f"eval/{label}", "evaluation", config=dict(policy=label),
                                 tags=["eval"] + label.split("/")[:2])
            levels = sorted(f for lab, f in data if lab == label)
            for f in levels:
                run.summary.update(tracking.flatten(per_policy[f"{label}@{f}"], prefix=f"level{f}"))
            run.log({"episodes": tracking.table([{k: v for k, v in r.items() if not isinstance(v, dict)}
                                                 for f in levels for r in data[(label, f)]])})
            run.finish()
        run = tracking.start(cfg, "eval/summary", "evaluation", tags=["eval", "summary"])
        table_rows = [dict(dataset=e["dataset"], factor=e["factor"], success_mean=e["success_mean"],
                           success_std=e["success_std"], seeds=len(e["seeds"]),
                           only_this=e.get("vs_base_only_this"), only_base=e.get("vs_base_only_base"),
                           sign_test_p=e.get("vs_base_sign_test_p")) for e in by_dataset.values()]
        log: dict[str, Any] = {"datasets": tracking.table(table_rows)} if table_rows else {}
        for factor in sorted({e["factor"] for e in by_dataset.values()}):
            data_rows = [[e["dataset"], e["success_mean"]] for e in by_dataset.values() if e["factor"] == factor]
            log[f"success_level{factor}"] = wandb.plot.bar(wandb.Table(data=data_rows, columns=["dataset", "success"]),
                                                           "dataset", "success", title=f"success, level {factor} (mean over seeds)")
        policy_rows = [dict(policy=v["label"], factor=v["factor"], success_rate=v["success_rate"], episodes=v["episodes"])
                       for v in per_policy.values()]
        if policy_rows:
            log["policies"] = tracking.table(policy_rows)
        run.summary.update(tracking.flatten({k: dict(mean=v["success_mean"], std=v["success_std"]) for k, v in by_dataset.items()}))
        run.log(log)
        run.finish()
    return summary
