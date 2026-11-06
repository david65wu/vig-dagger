"""Raw, uncalibrated flow-loss traces across OpenDrawer resolution levels.

The simulator worker uses one RNG-isolated stream per observation, shared across
scoring resolutions. Non-acting scores cannot change the executed action sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

FACTORS = (1, 2, 4, 8, 16, 32, 64)
GROUPS = ("moe", "per_resolution")
TITLES = {"moe": "MoE", "per_resolution": "Seven resolution-specific policies"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def query_seed(scene_seed, step):
    key = f"opendrawer-raw-flow-v1:{scene_seed}:{step}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31)


def score_query(policies, factors, state, images, acting_factor, seed, draws):
    """Each level scores its own generated chunk; actor executes that same chunk.

    Equal random seeds provide common noise/time draws across levels. Forking
    restores CPU/CUDA RNG state, including when a scoring call fails.
    """
    import torch

    losses, chunks, action = [], [], None
    for factor in factors:
        policy = policies[factor]
        devices = [policy.device.index if policy.device.index is not None else torch.cuda.current_device()] \
            if policy.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            value, samples = policy.flow_score(state, images, num_action_samples=1,
                                               num_loss_samples=draws, factor=factor)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite flow loss at resolution {factor}")
        losses.append(value)
        chunks.append(samples[0].detach().cpu().numpy())
        if factor == acting_factor:
            action = policy.denormalize_actions(samples[0])
    if action is None:
        raise ValueError("Acting factor must be among scoring factors")
    return np.asarray(losses, dtype=np.float64), np.stack(chunks), action


def episode_dir(cfg, group, acting_factor, seed):
    return Path(cfg["work_dir"]) / cfg.get("rollout_subdir", "rollouts") / group / f"acting_{acting_factor}" / str(seed)


def episode_done(cfg, group, acting_factor, seed):
    p = episode_dir(cfg, group, acting_factor, seed)
    try:
        meta = json.loads((p / "episode.json").read_text())
        return (meta["protocol_id"] == cfg["protocol_id"] and (p / "trace.npz").is_file()
                and (not cfg.get("record_video") or (p / "cameras.mp4").is_file()))
    except (OSError, ValueError, KeyError):
        return False


def build_jobs(cfg, config_path, pilot=False):
    from vig_dagger.jobs import Job
    from vig_dagger.plan import isaac_job

    jobs = []
    seeds = cfg["scene_seeds"][:1] if pilot else cfg["scene_seeds"]
    acting_factors = cfg["acting_factors"][:1] if pilot else cfg["acting_factors"]
    for factor in acting_factors:
        for group in cfg["groups"]:
            args = ["vig_dagger.workers.resolution_flow", "--config", str(config_path),
                    "--group", group, "--acting-factor", str(factor)]
            if pilot:
                args += ["--pilot"]
            job = Job(f"{'pilot' if pilot else 'rollout'}/{group}/acting_{factor}", args,
                      kind="sim", priority=10, retries=2,
                      done=lambda group=group, factor=factor: all(
                          episode_done(cfg, group, factor, seed) for seed in seeds))
            jobs.append(isaac_job(cfg, job))
    if not pilot:
        jobs.append(Job("raw_flow_report", ["vig_dagger.resolution_flow", "report", "--config", str(config_path)],
                        kind="cpu", deps=[j.name for j in jobs], priority=50,
                        done=lambda: report_done(cfg)))
    return jobs


def report_done(cfg):
    try:
        meta = json.loads((Path(cfg["work_dir"]) / "results/report.done.json").read_text())
        return meta["protocol_id"] == cfg["protocol_id"]
    except (OSError, ValueError, KeyError):
        return False


def records(cfg):
    out = []
    for group in cfg["groups"]:
        for factor in cfg["acting_factors"]:
            for seed in cfg["scene_seeds"]:
                if episode_done(cfg, group, factor, seed):
                    p = episode_dir(cfg, group, factor, seed)
                    row = json.loads((p / "episode.json").read_text())
                    with np.load(p / "trace.npz") as data:
                        flow = data["flow"].copy()
                        if flow.shape != (row["queries"], len(cfg["factors"])) or not np.isfinite(flow).all():
                            raise ValueError(f"Invalid flow trace: {p}")
                    out.append((row, flow, p))
    return out


def plot_episode(cfg, row, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = Path(cfg["work_dir"]) / "results/traces" / row["group"] / f"acting_{row['acting_factor']}"
    dest.mkdir(parents=True, exist_ok=True)
    png, pdf = dest / f"{row['seed']}.png", dest / f"{row['seed']}.pdf"
    if png.is_file() and pdf.is_file():
        return
    with np.load(path / "trace.npz") as data:
        flow, seconds, stages = data["flow"], data["query_seconds"], data["state_stages"]
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, factor in enumerate(cfg["factors"]):
        actor = factor == row["acting_factor"]
        ax.plot(seconds, flow[:, i], color=colors[i], linewidth=2.1 if actor else 1.1,
                alpha=1 if actor else .8, label=f"{factor}" + (" (acting)" if actor else ""))
    for i, label in enumerate(("Drawer open", "Charger picked", "Charger placed", "Drawer closed")):
        where = np.flatnonzero(stages[:, i])
        if len(where):
            t = where[0] * row["dt"]
            ax.axvline(t, color="grey", linestyle=":", alpha=.5)
            ax.text(t, .98 - .07 * (i % 2), label, transform=ax.get_xaxis_transform(),
                    fontsize=7, rotation=90, ha="right", va="top", color="#555555")
    ax.set(xlabel="Simulated time (seconds)", ylabel="Raw flow-matching loss",
           title=f"{TITLES[row['group']]} | acting factor {row['acting_factor']} | seed {row['seed']}\n"
                 f"{'SUCCESS' if row['success'] else 'FAILED'} | seven scores at the same observation")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(title="Scored resolution factor", ncols=4, fontsize=8, loc="upper center",
              bbox_to_anchor=(.5, -.18), frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def report(cfg):
    """Readable separate-group raw-value plots; partial reports state their N."""
    import csv
    import fcntl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(cfg["work_dir"])
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = records(cfg)
        for row, _, path in data:
            if cfg.get("record_video"):
                from vig_dagger.resolution_flow_video import render_episode
                render_episode(cfg, row, path)
            else:
                plot_episode(cfg, row, path)
        summary, metric_rows = {}, []
        for group in cfg["groups"]:
            subset = [(r, a) for r, a, _ in data if r["group"] == group]
            if not subset:
                continue
            dest = out / group
            dest.mkdir(exist_ok=True)
            pooled = np.concatenate([a for _, a in subset])
            means = np.stack([a.mean(axis=0) for _, a in subset])
            # Primary comparison: each resolution's loss on its own 20 rollouts.
            # Keep this distinct from scoring all levels on every visited state.
            own = [[a[:, cfg["factors"].index(factor)] for r, a in subset
                    if r["acting_factor"] == factor] for factor in cfg["acting_factors"]]
            for key, average in [("boxplot_own_rollouts", False), ("boxplot_own_rollout_means", True)]:
                values = [(np.asarray([a.mean() for a in arrays]) if average else np.concatenate(arrays))
                          if arrays else np.asarray([]) for arrays in own]
                fig, ax = plt.subplots(figsize=(8, 5.2))
                ax.boxplot(values, tick_labels=[f"{f}\nn={len(a)}" for f, a in zip(cfg["acting_factors"], own)],
                           showmeans=True, meanprops=dict(marker="D", markersize=4,
                               markerfacecolor="#d65f00", markeredgecolor="#d65f00"),
                           flierprops=dict(marker=".", markersize=2, alpha=.2))
                ax.set(xlabel="Acting and scored resolution factor (n = rollouts)",
                       ylabel="Mean raw flow loss per rollout" if average else "Raw flow-matching loss",
                       title=f"{TITLES[group]}: each resolution on its own rollouts\n"
                             + ("One point per rollout" if average else "All policy-query observations"))
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=.2)
                fig.text(.5, .015, "Box: 25–75%; line: median; diamond: mean; whiskers: 1.5×IQR; dots: outliers",
                         ha="center", fontsize=8)
                fig.tight_layout(rect=(0, .04, 1, 1))
                fig.savefig(dest / f"{key}.png", dpi=160)
                fig.savefig(dest / f"{key}.pdf")
                plt.close(fig)
            for key, values, title, label in [
                ("boxplot_queries", pooled, "All policy-query observations", "Raw flow-matching loss"),
                ("boxplot_rollout_means", means, "One mean per rollout", "Mean raw flow-matching loss per rollout"),
            ]:
                fig, ax = plt.subplots(figsize=(8, 5.2))
                boxes = ax.boxplot([values[:, i] for i in range(len(cfg["factors"]))],
                                  tick_labels=[str(f) for f in cfg["factors"]], showmeans=True,
                                  meanprops=dict(marker="D", markersize=4, markerfacecolor="#d65f00", markeredgecolor="#d65f00"),
                                  flierprops=dict(marker=".", markersize=2, alpha=.2))
                ax.set(xlabel="Scored resolution factor", ylabel=label,
                       title=f"{TITLES[group]}: {title}\n{len(subset)} rollouts; {len(pooled):,} query observations")
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=.2)
                fig.text(.5, .015, "Box: 25–75%; line: median; diamond: mean; whiskers: 1.5×IQR; dots: outliers",
                         ha="center", fontsize=8)
                fig.tight_layout(rect=(0, .04, 1, 1))
                fig.savefig(dest / f"{key}.png", dpi=160)
                fig.savefig(dest / f"{key}.pdf")
                plt.close(fig)
            # The pooled scores share observations, but the actor changes the
            # visited states. Also expose the comparison within each actor level.
            fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
            for ax, acting in zip(axes.flat, cfg["acting_factors"]):
                selected = [a for r, a in subset if r["acting_factor"] == acting]
                if selected:
                    values = np.concatenate(selected)
                    ax.boxplot([values[:, i] for i in range(len(cfg["factors"]))],
                               tick_labels=[str(f) for f in cfg["factors"]], showmeans=True,
                               flierprops=dict(marker=".", markersize=1, alpha=.2))
                else:
                    ax.text(.5, .5, "Pending", transform=ax.transAxes, ha="center")
                ax.set(title=f"Acting factor {acting} | n={len(selected)} rollouts",
                       xlabel="Scored resolution factor", ylabel="Raw flow-matching loss")
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=.2)
            for ax in list(axes.flat)[len(cfg["acting_factors"]):]:
                ax.set_visible(False)
            fig.suptitle(f"{TITLES[group]}: raw query losses, separated by acting resolution")
            fig.tight_layout(rect=(0, 0, 1, .95))
            fig.savefig(dest / "boxplot_by_acting_resolution.png", dpi=160)
            fig.savefig(dest / "boxplot_by_acting_resolution.pdf")
            plt.close(fig)
            summary[group] = dict(rollouts=len(subset), queries=len(pooled),
                                  successes=sum(r["success"] for r, _ in subset), levels={})
            for i, factor in enumerate(cfg["factors"]):
                stats = dict(queries=len(pooled), rollouts=len(subset),
                             mean=float(pooled[:, i].mean()), median=float(np.median(pooled[:, i])),
                             q25=float(np.percentile(pooled[:, i], 25)), q75=float(np.percentile(pooled[:, i], 75)),
                             rollout_mean_average=float(means[:, i].mean()))
                summary[group]["levels"][str(factor)] = stats
                metric_rows.append(dict(group=group, scored_factor=factor, **stats))
        expected = len(cfg["groups"]) * len(cfg["acting_factors"]) * len(cfg["scene_seeds"])
        write_json(out / "summary.json", dict(protocol_id=cfg["protocol_id"], completed=len(data),
                                              expected=expected, groups=summary))
        if metric_rows:
            with (out / "flow_statistics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
                writer.writeheader()
                writer.writerows(metric_rows)
        lines = ["# Raw flow loss across resolutions", "", f"Completed rollouts: **{len(data)}/{expected}**.", "",
                 "No calibration, thresholds, or rescaling of the recorded losses is applied. Each score is the mean "
                 f"flow-matching loss of one policy-generated 16-action chunk over {cfg['loss_draws']} noise/time draws. "
                 "It is generated-action self-consistency, not error against an expert action. Losses use each checkpoint's "
                 "own trained action normalization. Higher raw values are not calibrated failure probabilities.", "",
                 f"Scores are measured before each policy query, every {cfg['actions_per_step']} simulator steps. "
                 "All seven scoring levels see the same current observation. The acting level is fixed within a rollout "
                 "and executes its scored chunk's first eight actions. Common random draws are used across levels; "
                 "scoring restores simulator RNG state. No expert intervenes. Failed rollouts are included.", "",
                 "The pooled-query box plot gives each query equal weight, so longer rollouts contribute more points. "
                 "The rollout-mean box plot gives each rollout one point per scoring level. The actor-separated panels "
                 "show how the visited states affect the comparison. All plots use linear axes and raw loss units.", ""]
        if cfg.get("record_video"):
            lines += ["Rollout MP4s show the original left/wrist RGB images above the seven raw loss curves. "
                      "The moving cursor and camera frame use the same simulation-step timestamp. Camera frames are "
                      "recorded every two steps (30 fps); loss scores every eight steps. Success and failure videos "
                      "are in separate folders. Earlier attempts without images remain archived under `../rollouts/` "
                      "and are not included in this report.", ""]
        for group in cfg["groups"]:
            if group not in summary:
                lines += [f"## {TITLES[group]}", "", "Waiting for the first completed rollout.", ""]
                continue
            lines += [f"## {TITLES[group]}", "",
                      "### Each resolution on its own rollouts", "",
                      "These are the direct rollout comparisons: each box uses the acting level's own loss, "
                      "from its 20 attempts. States visited can differ between resolutions.", "",
                      f"![Own rollout losses]({group}/boxplot_own_rollouts.png)", "",
                      f"![Own rollout means]({group}/boxplot_own_rollout_means.png)", "",
                      "### All seven scoring levels on shared observations", "",
                      "Each box below scores the same pooled observations from all acting resolutions.", "",
                      f"![Query losses]({group}/boxplot_queries.png)", "",
                      f"![Rollout means]({group}/boxplot_rollout_means.png)", "",
                      f"[Box plots by acting resolution]({group}/boxplot_by_acting_resolution.png)", "",
                      "| Acting factor | Seed | Outcome | Queries | Rollout |", "|---|---|---|---:|---|"]
            for row, _, _ in data:
                if row["group"] == group:
                    if cfg.get("record_video"):
                        outcome = "success" if row['success'] else "failed"
                        rel = f"videos/{group}/acting_{row['acting_factor']}/{outcome}/{row['seed']}.mp4"
                        label = "MP4"
                    else:
                        rel = f"traces/{group}/acting_{row['acting_factor']}/{row['seed']}.png"
                        label = "Plot"
                    lines.append(f"| {row['acting_factor']} | {row['seed']} | {'Success' if row['success'] else 'Failed'} "
                                 f"| {row['queries']} | [{label}]({rel}) |")
            lines.append("")
        (out / "REPORT.md").write_text("\n".join(lines))
        if len(data) == expected:
            write_json(out / "report.done.json", dict(protocol_id=cfg["protocol_id"], rollouts=len(data), time=time.time()))
        return len(data), expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "report", "watch"))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.command == "run":
        from vig_dagger.jobs import Queue, lock
        root = Path(cfg["work_dir"])
        lock(root)
        jobs = build_jobs(cfg, args.config.resolve(), args.pilot)
        _, failed = Queue(jobs, cfg["resources"], root / "logs").run()
        if failed:
            raise SystemExit(f"Failed jobs: {sorted(failed)}")
    elif args.command == "report":
        print("REPORT", report(cfg), flush=True)
    else:
        previous = -1
        while True:
            count = len(records(cfg))
            if count != previous:
                done, expected = report(cfg)
                print(f"[raw-flow] rendered {done}/{expected} rollouts", flush=True)
                previous = count
                if done == expected:
                    break
            time.sleep(30)


if __name__ == "__main__":
    main()
