"""Threshold-sweep report: one Markdown file with every number the threshold analysis needs.

For a sweep-mode experiment (fixed thresholds, one MoE), ``vig-dagger sweep-report -c <config>`` writes
``results/SWEEP_REPORT.md``: readable tables (the collections, the dense policies, paired tests, what the threshold
rules say at each threshold) and, at the end, one JSON block with the data behind three figures and behind the
threshold analysis. A machine that can only send a text file back sends this one file; anywhere,

    python -m vig_dagger.sweep_report plot SWEEP_REPORT.md --out <dir>

redraws from the file alone ``collections.png`` (success per attempt and expert shares per threshold, drawn by
``summary.figure``), ``retraining.png`` (success of the dense policies, the MoE and any extra policy such as a
planner-only one, drawn by ``evaluation.figure``) and ``distributions.png`` (per threshold: VIG histograms of the MoE's
reference frames, the policy steps, policy + expert steps and expert steps).

Several machines with tasks assigned by hand (``vig-dagger run --tasks``, no board) each write their own report;

    python -m vig_dagger.sweep_report merge A.md B.md ... --out SWEEP_REPORT.md

joins them into one: every collection, training and evaluation, the paired comparisons recomputed from each scene's
result (``merge``).

The JSON block (numbers only: no images, no model weights):

  run          the config values the figures use, code version, GPUs, the MoE (source, checkpoint hash, training)
  reference    the MoE's VIG on its reference frames (every ``thresholds.frame_stride``-th frame of the first
               ``thresholds.num_demos`` planner demos): percentiles 0-100 of all frames and of the frames before each
               demo's success (a fast demo holds still for ~35 steps after it), means, per task stage
  collections  per threshold: the collection's numbers as the collections figure draws them; counts; the step
               histograms in the reference's percentile bins (policy steps split into counted / the scores that
               triggered a take-over / a failed attempt's last policy stretch, and expert steps; successful and failed
               attempts apart); each attempt's policy-step mean as the adaptive rule sees it (histograms and exact
               means, plain and with the failed-attempt rule, equal and step weights); distances of the step
               distributions to the reference; per-stage summaries; episode lengths; take-overs per attempt; the
               figure's 60-bin histograms
  retraining   per dense policy, the MoE and each extra policy: success, one 0/1 character per evaluation scene,
               training numbers; the evaluation summary the retraining figure draws

``merge-reference`` (``python -m vig_dagger.sweep_report merge-reference --config <work_dir>/config.json --seed S``)
merges scored reference parts into ``demo_scores.npz`` without touching ``thresholds.json``, for a MoE that was
downloaded with its thresholds (its reference frames then have to be scored here first).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from vig_dagger.adaptive import percentile_of
from vig_dagger.config import Collection, Paths, collections, downloaded, iterating

VERSION = 1
MARK = "<!-- sweep-report-data v1 -->"
OVERFLOW = [float(x) for x in np.geomspace(1.0, 8.0, 21)[1:]]  # bin edges above the reference's maximum, as multiples
LENGTH_BINS = list(range(0, 401, 20))  # episode length bins (the last one: 400 and more)
SEGMENT_BINS = list(range(0, 201, 10))  # expert-segment length bins (steps from a take-over to the hand-back or the end)
COARSE = list(range(0, 101, 10))  # 10-percentile bins (per stage, per episode progress); a last bin above p100
PROGRESS = 10  # episode progress deciles
CATEGORIES = {  # step histograms per collection, in the reference's percentile bins
    "policy_counted_success": "policy steps of successful attempts, as the threshold rules count them (without the "
                              "scores that triggered a take-over)",
    "policy_counted_failed_before_last": "policy steps of failed attempts before their last take-over (counted)",
    "policy_last_stretch_failed": "policy steps of failed attempts after their last take-over, or all of them when "
                                  "the expert never took over (counted; the failed-attempt rule sets them to the threshold)",
    "policy_trigger_success": "the scores that triggered each take-over (dagger.take_count per take-over), successful "
                              "attempts",
    "policy_trigger_failed": "the same, failed attempts",
    "expert_success": "scores after expert steps (the hand-back checks), successful attempts",
    "expert_failed": "the same, failed attempts",
}


# ---------------------------------------------------------------------------------------------------- small helpers
def _round(x, sig: int = 6):
    """Floats to ``sig`` significant digits, NaN/inf to None, numpy to Python; recursively."""
    if isinstance(x, dict):
        return {str(k): _round(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, sig) for v in x]
    if isinstance(x, np.ndarray):
        return [_round(v, sig) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not math.isfinite(x) else float(f"{x:.{sig}g}")
    return x


def _hist(values: np.ndarray, edges: np.ndarray) -> list[int]:
    """Counts in [edges[i], edges[i+1]); values below the first edge go to the first bin, above the last to the last."""
    if len(values) == 0:
        return [0] * (len(edges) - 1)
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1).astype(int).tolist()


def _ks(a: np.ndarray, b_sorted: np.ndarray) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic (b sorted)."""
    if len(a) == 0 or len(b_sorted) == 0:
        return None
    a = np.sort(a)
    grid = np.r_[a, b_sorted]
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b_sorted, grid, side="right") / len(b_sorted)
    return float(np.max(np.abs(fa - fb)))


def _w1(a: np.ndarray, b_sorted: np.ndarray) -> float | None:
    """1-Wasserstein distance between two samples (b sorted), in VIG units."""
    if len(a) == 0 or len(b_sorted) == 0:
        return None
    a = np.sort(a)
    grid = np.unique(np.r_[a, b_sorted])
    fa = np.searchsorted(a, grid[:-1], side="right") / len(a)
    fb = np.searchsorted(b_sorted, grid[:-1], side="right") / len(b_sorted)
    return float(np.sum(np.abs(fa - fb) * np.diff(grid)))


def _stats(values: np.ndarray, ref: np.ndarray, pre: np.ndarray, take: float) -> dict:
    """One set of step scores against the reference (all frames, and frames before success)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return dict(n=0)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), mean_pct=percentile_of(ref, float(v.mean())),
                median_pct=percentile_of(ref, float(np.median(v))), above_threshold=float((v > take).mean()),
                shift_vs_all=float(v.mean() - ref.mean()), ks_vs_all=_ks(v, ref), w1_vs_all=_w1(v, ref),
                shift_vs_pre_success=float(v.mean() - pre.mean()), ks_vs_pre_success=_ks(v, pre),
                w1_vs_pre_success=_w1(v, pre))


def _sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _code_version() -> dict:
    repo = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20)
        branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                                text=True, timeout=20)
        return dict(commit=head.stdout.strip() or None, branch=branch.stdout.strip() or None,
                    modified_files=[line[3:] for line in dirty.stdout.splitlines()])
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None)


def _gpus() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _pct_label(spec: str) -> float:
    return float(str(spec)[1:])


# ----------------------------------------------------------------------------------------------------- the reference
def reference_data(cfg: dict, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """(the reference section, all reference values sorted, the values before each demo's success sorted)."""
    import os

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py

    from vig_dagger import stages
    from vig_dagger.workers.score_demos import demo_list

    paths = Paths.of(cfg)
    path = paths.demo_scores(seed)
    if not path.is_file():
        raise SystemExit(f"{path} is missing: the MoE's reference frames are not scored here (a downloaded MoE: score "
                         "them with workers/score_demos.py, then `python -m vig_dagger.sweep_report merge-reference`)")
    z = np.load(path)
    vig, demo, frame = z["vig"].astype(float), z["demo"].astype(int), z["frame"].astype(int)
    listed = demo_list(cfg)
    after = np.zeros(len(vig), bool)
    stage = np.zeros(len(vig), int)
    by_part: dict[Path, list[tuple[int, int]]] = {}
    for d in np.unique(demo):
        part, i = listed[int(d)]
        by_part.setdefault(part, []).append((int(d), int(i)))
    lengths, holding = [], []
    progress = np.zeros(len(vig))  # frame / demo length
    for part, items in by_part.items():
        with h5py.File(part, "r") as f:
            for d, i in items:
                g = f[f"traj_{i}"]
                success = np.asarray(g["success"][:], bool)
                first = int(np.argmax(success)) if success.any() else None
                ts = g["task_state"]
                state = {k: ts[k][:] for k in ("goal_pose", "charger_pose", "tcp_pose", "held", "dist", "angle")}
                lab, _ = stages.label(state, bool(success.any()))
                sel = demo == d
                fr = frame[sel]
                after[sel] = fr >= first + 1 if first is not None else False
                stage[sel] = lab[np.minimum(fr, len(lab) - 1)]
                progress[sel] = fr / max(len(success), 1)
                lengths.append(len(success))
                holding.append(len(success) - first - 1 if first is not None else 0)
    ref, pre = np.sort(vig), np.sort(vig[~after])
    q = np.arange(101)
    by_stage = {}
    for s, name in enumerate(stages.STAGES):
        sel = stage == s
        if sel.any():
            by_stage[name] = dict(frames=int(sel.sum()), holding_frames=int((sel & after).sum()), mean=float(vig[sel].mean()),
                                  mean_pct=percentile_of(ref, float(vig[sel].mean())))
    coarse = coarse_edges(ref)
    stage_hist = {name: _hist(vig[stage == s], coarse) for s, name in enumerate(stages.STAGES) if (stage == s).any()}
    dec = np.minimum((progress * PROGRESS).astype(int), PROGRESS - 1)
    by_progress = [[int((dec == k).sum()), percentile_of(ref, float(vig[dec == k].mean())) if (dec == k).any() else None,
                    float(after[dec == k].mean()) if (dec == k).any() else None] for k in range(PROGRESS)]
    t = cfg["thresholds"]
    out = dict(frames=len(vig), demos=int(len(np.unique(demo))), frame_stride=int(t["frame_stride"]),
               first_demos=int(t["num_demos"]), demo_steps_mean=float(np.mean(lengths)),
               holding_steps_mean=float(np.mean(holding)), holding_frame_share=float(after.mean()),
               percentiles_all=np.percentile(ref, q), percentiles_pre_success=np.percentile(pre, q),
               mean_all=float(ref.mean()), mean_all_pct=percentile_of(ref, float(ref.mean())),
               mean_pre_success=float(pre.mean()), mean_pre_success_pct=percentile_of(ref, float(pre.mean())),
               mean_pre_success_pct_of_pre=percentile_of(pre, float(pre.mean())), by_stage=by_stage,
               stage_histograms=stage_hist,
               by_progress=dict(columns=["frames", "mean VIG as a percentile", "share after success"], deciles=by_progress))
    return out, ref, pre


def bin_edges(ref: np.ndarray) -> np.ndarray:
    """The step histograms' bins: the reference's 100 percentile bins, then 20 bins up to 8x its maximum, then one to
    infinity."""
    inner = np.percentile(ref, np.arange(101))
    return np.r_[inner, inner[-1] * np.asarray(OVERFLOW), np.inf]


def coarse_edges(ref: np.ndarray) -> np.ndarray:
    """10-percentile bins of the reference, then one bin above its maximum."""
    return np.r_[np.percentile(ref, COARSE), np.inf]


# ----------------------------------------------------------------------------------------------------- a collection
def collection_data(cfg: dict, col: Collection, ref: np.ndarray, pre: np.ndarray, edges: np.ndarray,
                    target_pct: float) -> dict | None:
    import os

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py

    from vig_dagger import stages
    from vig_dagger.recording import EXPERT, MOVE, POLICY

    paths = Paths.of(cfg)
    out = paths.dagger(col)
    try:
        settings = json.loads((out / "settings.json").read_text())
    except (OSError, ValueError):
        return None
    take, back = float(settings["take_threshold"]), float(settings.get("return_threshold", settings["take_threshold"]))
    take_count = int(settings["take_count"])
    summaries = {}
    for path in sorted((out / "episodes").glob("worker_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                summaries[row["seed"]] = row
    cats = {k: [] for k in CATEGORIES}
    sets = {k: [] for k in ("policy_counted", "policy_all", "expert", "policy_counted_success", "policy_counted_failed")}
    attempts = []  # per attempt: success, finished_at, n counted, mean, mean with the failed-attempt rule
    n_stages = len(stages.STAGES)
    coarse = coarse_edges(ref)
    stage_vals = {"policy_counted": [[] for _ in range(n_stages)], "expert": [[] for _ in range(n_stages)]}
    prog_vals = [[] for _ in range(PROGRESS)]  # counted policy-step scores by episode progress decile
    take_stage = [0] * n_stages
    segments = []
    times = []
    stage_policy = [[0, 0.0] for _ in range(n_stages)]  # count, sum of percentiles
    stage_expert = [[0, 0.0] for _ in range(n_stages)]
    failed_at = [0] * n_stages
    lengths = {True: [], False: []}
    takeovers = {True: [0] * 11, False: [0] * 11}
    steps_by = {k: 0 for k in ("policy", "expert", "move")}
    for path in sorted((out / "attempts").glob("worker_*.h5")):
        with h5py.File(path, "r") as f:
            for name in f:
                g = f[name]
                s = summaries.get(int(g.attrs["seed"]))
                if s is None:  # interrupted before its summary line
                    continue
                success = bool(s["success"])
                score = np.asarray(g["score"][:], float)
                controller = np.asarray(g["controller"][:], int)
                take_steps = [int(t) for t in g["take_steps"][:]]
                came = np.r_[POLICY, controller][: len(score)]  # observation t is scored after step t-1
                finite = np.isfinite(score)
                pol = np.flatnonzero(finite & (came != EXPERT))
                place = {int(t): i for i, t in enumerate(pol)}
                trig = set()
                for t in take_steps:
                    i = place.get(t)
                    if i is not None:
                        trig.update(pol[max(0, i - take_count + 1): i + 1].tolist())
                counted = np.array([t for t in pol if t not in trig], int)
                trig_idx = np.array(sorted(trig), int)
                exp_idx = np.flatnonzero(finite & (came == EXPERT))
                last_from = max(take_steps) if take_steps else -1
                c_scores = score[counted]
                if success:
                    cats["policy_counted_success"].append(c_scores)
                    cats["policy_trigger_success"].append(score[trig_idx])
                    cats["expert_success"].append(score[exp_idx])
                    sets["policy_counted_success"].append(c_scores)
                    last_n = 0
                else:
                    early = counted <= last_from
                    cats["policy_counted_failed_before_last"].append(score[counted[early]])
                    cats["policy_last_stretch_failed"].append(score[counted[~early]])
                    cats["policy_trigger_failed"].append(score[trig_idx])
                    cats["expert_failed"].append(score[exp_idx])
                    sets["policy_counted_failed"].append(c_scores)
                    last_n = int((~early).sum())
                sets["policy_counted"].append(c_scores)
                sets["policy_all"].append(score[pol])
                sets["expert"].append(score[exp_idx])
                n = len(c_scores)
                mean = float(c_scores.mean()) if n else None
                ruled = (np.r_[c_scores[: n - last_n], np.full(last_n, take)] if last_n else c_scores)
                attempts.append(dict(success=success, order=s.get("finished_at", s["seed"]), n=n, mean=mean,
                                     mean_failed_rule=float(ruled.mean()) if n else None))
                state = {k: g[k][:] for k in ("goal_pose", "charger_pose", "tcp_pose", "held", "dist", "angle")}
                stage, flags = stages.label(state, success)
                for idx, acc, key in ((counted, stage_policy, "policy_counted"), (exp_idx, stage_expert, "expert")):
                    for t in idx:
                        st = int(stage[min(int(t), len(stage) - 1)])
                        acc[st][0] += 1
                        acc[st][1] += percentile_of(ref, float(score[t]))
                        stage_vals[key][st].append(float(score[t]))
                for t in counted:
                    prog_vals[min(int(PROGRESS * t / max(len(controller), 1)), PROGRESS - 1)].append(float(score[t]))
                for t in take_steps:
                    take_stage[int(stage[min(t, len(stage) - 1)])] += 1
                returns = sorted(int(t) for t in g["return_steps"][:])
                for t in take_steps:
                    end = next((r for r in returns if r > t), len(controller))
                    segments.append(end - t)
                if s.get("finished_at") is not None:
                    times.append(float(s["finished_at"]))
                if not success:
                    failed_at[stages.STAGES.index(stages.summary(stage, flags, success)["failed_at"])] += 1
                lengths[success].append(len(controller))
                takeovers[success][min(len(take_steps), 10)] += 1
                steps_by["policy"] += int((controller == POLICY).sum())
                steps_by["expert"] += int((controller == EXPERT).sum())
                steps_by["move"] += int((controller == MOVE).sum())
    if not attempts:
        return None
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cats.items()}
    st = {k: np.concatenate(v) if v else np.zeros(0) for k, v in sets.items()}
    # the attempts as the adaptive threshold sees them: q(m) of each attempt's counted policy-step mean
    rows = sorted([a for a in attempts if a["n"] > 0], key=lambda a: a["order"])
    q_plain = np.array([percentile_of(ref, a["mean"]) for a in rows])
    q_rule = np.array([percentile_of(ref, a["mean_failed_rule"]) for a in rows])
    n_arr = np.array([a["n"] for a in rows], float)
    ok = np.array([a["success"] for a in rows], bool)
    half = len(rows) // 2

    def means(q: np.ndarray) -> dict:
        if len(q) == 0:
            return {}
        return dict(equal=float(q.mean()), equal_se=float(q.std(ddof=1) / math.sqrt(len(q))) if len(q) > 1 else None,
                    steps=float((q * n_arr).sum() / n_arr.sum()),
                    success_equal=float(q[ok].mean()) if ok.any() else None,
                    failed_equal=float(q[~ok].mean()) if (~ok).any() else None,
                    first_half_equal=float(q[:half].mean()) if half else None,
                    second_half_equal=float(q[half:].mean()) if len(q) > half else None)

    qbins = np.linspace(0, 100, 101)
    qbins[-1] = 100.0 + 1e-9
    adaptive = dict(
        target_pct=target_pct, attempts_with_policy_steps=len(rows), plain=means(q_plain), failed_rule=means(q_rule),
        hist_bins="1-percentile bins 0-100 of q(m)",
        hist_plain_success=_hist(q_plain[ok], qbins), hist_plain_failed=_hist(q_plain[~ok], qbins),
        hist_failed_rule_failed=_hist(q_rule[~ok], qbins),
        hist_steps_plain_success=np.bincount(np.clip(np.searchsorted(qbins, q_plain[ok], "right") - 1, 0, 99),
                                             weights=n_arr[ok], minlength=100).astype(int).tolist(),
        hist_steps_plain_failed=np.bincount(np.clip(np.searchsorted(qbins, q_plain[~ok], "right") - 1, 0, 99),
                                            weights=n_arr[~ok], minlength=100).astype(int).tolist(),
        hist_steps_failed_rule_failed=np.bincount(np.clip(np.searchsorted(qbins, q_rule[~ok], "right") - 1, 0, 99),
                                                  weights=n_arr[~ok], minlength=100).astype(int).tolist())
    # the distributions figure's 60-bin histograms (as iterate_report.figure_distributions draws them)
    pol, exp = st["policy_counted"], st["expert"]
    both = np.r_[pol, exp]
    right = 1.1 * max(float(np.percentile(ref, 99.5)), float(np.percentile(both, 99.5)) if len(both) else 0.0)
    b60 = np.linspace(0.0, right, 61)
    clip = lambda x: np.clip(x, b60[0], b60[-1] - 1e-12)  # noqa: E731
    h = {k: np.histogram(clip(v), b60)[0].astype(int) for k, v in (("reference", ref), ("reference_pre_success", pre),
                                                                   ("policy", pol), ("both", both), ("expert", exp))}
    norm = {k: v / max(v.sum(), 1) for k, v in h.items()}
    figure60 = dict(right=right, bins=60, counts={k: v.tolist() for k, v in h.items()},
                    overlap=float(np.minimum(norm["policy"], norm["reference"]).sum()) if len(pol) else None,
                    overlap_both=float(np.minimum(norm["both"], norm["reference"]).sum()) if len(both) else None,
                    policy_above=float((pol > take).mean()) if len(pol) else None,
                    expert_above=float((exp > take).mean()) if len(exp) else None,
                    policy_mean=float(pol.mean()) if len(pol) else None,
                    policy_mean_pct=percentile_of(ref, float(pol.mean())) if len(pol) else None,
                    expert_mean_pct=percentile_of(ref, float(exp.mean())) if len(exp) else None)
    n_ok = sum(a["success"] for a in attempts)
    return dict(
        id=col.id, threshold=col.threshold, threshold_pct=_pct_label(col.threshold), threshold_vig=take,
        return_threshold=col.return_spec, return_vig=back, take_count=take_count,
        return_count=int(settings["return_count"]), target=int(settings["target"]), attempts=len(attempts),
        successes=n_ok, steps=steps_by,
        step_histograms={k: _hist(v, edges) for k, v in cat.items()},
        stats={k: _stats(v, ref, pre, take) for k, v in st.items()},
        adaptive=adaptive,
        by_stage=dict(policy_counted=[[c, (s / c) if c else None] for c, s in stage_policy],
                      expert=[[c, (s / c) if c else None] for c, s in stage_expert], failed_at=failed_at),
        stage_histograms={k: [_hist(np.asarray(v, float), coarse) for v in vals] for k, vals in stage_vals.items()},
        by_progress=[[len(v), percentile_of(ref, float(np.mean(v))) if v else None] for v in prog_vals],
        takeover_stage=take_stage,
        expert_segments=dict(bins=SEGMENT_BINS, counts=_hist(np.asarray(segments, float), np.r_[SEGMENT_BINS, np.inf])),
        hours=(max(times) - min(times)) / 3600 if len(times) > 1 else None,
        lengths=dict(bins=LENGTH_BINS, success=_hist(np.asarray(lengths[True], float), np.r_[LENGTH_BINS, np.inf]),
                     failed=_hist(np.asarray(lengths[False], float), np.r_[LENGTH_BINS, np.inf])),
        takeovers_per_attempt=dict(success=takeovers[True], failed=takeovers[False]),
        figure=figure60)


# ------------------------------------------------------------------------------------------------- the whole report
def _scene_line(rows: list[dict], first: int, n: int) -> str:
    by = {r["seed"]: r["success"] for r in rows}
    return "".join("-" if s not in by else ("1" if by[s] else "0") for s in range(first, first + n))


def build(cfg: dict) -> dict:
    from vig_dagger import dataset, evaluation, report, summary

    if iterating(cfg):
        raise SystemExit("sweep-report is for sweep mode (fixed thresholds); iteration mode has `vig-dagger report`")
    seeds = [int(s) for s in cfg["moe"]["seeds"]]
    if len(seeds) != 1:
        raise SystemExit("sweep-report covers one MoE (moe.seeds with one seed)")
    seed = seeds[0]
    paths = Paths.of(cfg)
    ref_data, ref, pre = reference_data(cfg, seed)
    edges = bin_edges(ref)
    thresholds = json.loads(paths.thresholds(seed).read_text())
    cols = [c for c in collections(cfg) if c.seed == seed]
    col_data = [d for d in (collection_data(cfg, c, ref, pre, edges, ref_data["mean_all_pct"]) for c in cols) if d]
    rows = summary.collection_rows(cfg)
    settings = summary.settings_rows(cfg, rows)
    ev = evaluation.summarize(cfg)
    eval_rows = evaluation.load(paths.root / "eval")
    first, n_eval = int(cfg["evaluation"]["seed_start"]), int(cfg["evaluation"]["num_episodes"])
    policies = {}
    for label, rws in sorted(eval_rows.items()):
        rws = [r for r in rws if first <= r["seed"] < first + n_eval]
        succ = [r for r in rws if r["success"]]
        policies[label] = dict(episodes=len(rws), successes=len(succ), scenes=_scene_line(rws, first, n_eval),
                               success_steps_median=float(np.median([r["steps"] for r in succ])) if succ else None,
                               failed_at={s: sum(r["failed_at"] == s for r in rws if not r["success"])
                                          for s in sorted({r["failed_at"] for r in rws if not r["success"]})})
    trainings = {r["run"]: r for r in report.training_summary(cfg)}
    datasets = {}
    for c in cols:
        root = paths.dataset(f"base_plus_{c.id}")
        if dataset.ready(root):
            datasets[c.id] = dict(episodes=dataset.num_demos(root), cut=dataset.cut_stats(root))
    for name in ("base",):
        if dataset.ready(paths.dataset(name)):
            datasets[name] = dict(episodes=dataset.num_demos(paths.dataset(name)))
    for label, run in report.training_runs(cfg):
        try:
            info = json.loads(((run / "run_config.json") if (run / "run_config.json").exists()
                               else (run / "train_config.json")).read_text()).get("_info", {})
        except (OSError, ValueError):
            info = {}
        trainings.setdefault(label, {}).update(info=info)
    moe_ckpt = paths.moe_checkpoint(seed)
    keep_cfg = dict(name=cfg["name"], dagger=cfg["dagger"], moe=dict(seeds=seeds, factors=cfg["moe"]["factors"]),
                    demos=dict(num_demos=cfg["demos"]["num_demos"], motion=cfg["demos"]["motion"],
                               seed_start=cfg["demos"]["seed_start"]),
                    retrain=dict(cut_random_moves=cfg["retrain"]["cut_random_moves"], seeds=cfg["retrain"]["seeds"]),
                    training=dict(epochs=cfg["training"]["epochs"]),
                    evaluation={k: cfg["evaluation"][k] for k in ("seed_start", "num_episodes", "max_episode_steps")},
                    thresholds={k: cfg["thresholds"][k] for k in ("num_demos", "frame_stride")},
                    extra_policies=cfg["evaluation"].get("extra_policies", {}))
    eval_fig = {k: ev.get(k) for k in ("episodes_per_policy", "scenes", "moes", "moe_counts", "settings", "extra")}
    eval_fig["retrained"] = [{k: v for k, v in r.items()} for r in ev["retrained"]]
    eval_fig["references"] = ev.get("references", [])
    return _round(dict(
        version=VERSION, created=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        run=dict(config=keep_cfg, code=_code_version(), gpus=_gpus(),
                 moe=dict(seed=seed, source=downloaded(cfg, seed) or "trained here", checkpoint_sha256=_sha256(moe_ckpt),
                          thresholds={k: v for k, v in thresholds.get("vig", {}).items()},
                          thresholds_frames=thresholds.get("frames"))),
        reference=ref_data, bin_edges=edges[:-1], collections=col_data,
        collections_figure=dict(settings=settings, references=[], rows=rows),
        retraining=dict(policies=policies, trainings=trainings, datasets=datasets, figure=eval_fig)))


# ------------------------------------------------------------------------------------------------------- the analysis
def analysis(data: dict, resamples: int = 2000, rng_seed: int = 0) -> dict:
    """What the report's tables say, from the data alone (so it can be recomputed from any report)."""
    cols = sorted(data["collections"], key=lambda c: c["threshold_pct"])
    pol = data["retraining"]["policies"]
    seed = data["run"]["moe"]["seed"]
    rt = data["run"]["config"]["retrain"]["seeds"][0]
    lines = {c["threshold"]: pol.get(f"retrain/{c['id']}/seed{rt}", {}).get("scenes") for c in cols}
    moe_line = pol.get(f"moe/sd{seed}", {}).get("scenes")
    extra = {k[len("extra/"):]: v["scenes"] for k, v in pol.items() if k.startswith("extra/")}

    def arr(line):
        return None if not line or "-" in line else np.array([ch == "1" for ch in line], bool)

    def paired(a, b):
        a, b = arr(a), arr(b)
        if a is None or b is None:
            return None
        w, l = int((a & ~b).sum()), int((b & ~a).sum())
        return dict(only_first=w, only_second=l, p=_sign_test(w, l))

    out = dict(thresholds=[c["threshold"] for c in cols], dense={}, paired={}, best={}, rules={})
    done = [c for c in cols if arr(lines[c["threshold"]]) is not None]
    for c in cols:
        a = arr(lines[c["threshold"]])
        if a is not None:
            lo, hi = _wilson(int(a.sum()), len(a))
            out["dense"][c["threshold"]] = dict(successes=int(a.sum()), episodes=len(a), rate=float(a.mean()), low=lo, high=hi)
    for x, y in zip(done, done[1:]):
        out["paired"][f"{y['threshold']} vs {x['threshold']}"] = paired(lines[y["threshold"]], lines[x["threshold"]])
    for c in done:
        if moe_line:
            out["paired"][f"{c['threshold']} vs MoE"] = paired(lines[c["threshold"]], moe_line)
        for name, line in extra.items():
            out["paired"][f"{c['threshold']} vs {name}"] = paired(lines[c["threshold"]], line)
    if len(done) >= 3:
        mat = np.stack([arr(lines[c["threshold"]]) for c in done]).astype(float)  # thresholds x scenes
        pcts = np.array([c["threshold_pct"] for c in done])
        rng = np.random.default_rng(rng_seed)
        raw_best, vertex = [], []
        for _ in range(resamples):
            idx = rng.integers(0, mat.shape[1], mat.shape[1])
            rates = mat[:, idx].mean(1)
            raw_best.append(pcts[int(np.argmax(rates))])
            b2, b1, b0 = np.polyfit(pcts, rates, 2)
            # the fitted curve's highest point within the swept range (the vertex when it is a concave peak inside)
            vertex.append(float(np.clip(-b1 / (2 * b2), pcts[0], pcts[-1])) if b2 < 0 else
                          float(pcts[int(np.argmax(np.polyval([b2, b1, b0], pcts)))]))
        rates = mat.mean(1)
        a2, a1, a0 = np.polyfit(pcts, rates, 2)
        out["best"] = dict(
            raw_best=[c["threshold"] for c, r in zip(done, rates) if r == rates.max()],
            quadratic=dict(coefficients=[a2, a1, a0], vertex=float(-a1 / (2 * a2)) if a2 != 0 else None,
                           concave=bool(a2 < 0)),
            resampled_scenes=resamples,
            raw_best_share={c["threshold"]: float(np.mean(np.array(raw_best) == p)) for c, p in zip(done, pcts)},
            quadratic_peak_median=float(np.median(vertex)),
            quadratic_peak_80=[float(np.percentile(vertex, 10)), float(np.percentile(vertex, 90))])
    # each rule as "how far the threshold should move" at each fixed threshold (positive: raise it), and where it
    # crosses zero
    target = data["reference"]["mean_all_pct"]
    rules = {
        "adaptive (attempts weigh the same, failed-attempt rule)": lambda c: target - c["adaptive"]["failed_rule"].get("equal", np.nan),
        "adaptive (policy steps weigh, failed-attempt rule)": lambda c: target - c["adaptive"]["failed_rule"].get("steps", np.nan),
        "adaptive (attempts weigh the same, no failed-attempt rule)": lambda c: target - c["adaptive"]["plain"].get("equal", np.nan),
        "adaptive (policy steps weigh, no failed-attempt rule)": lambda c: target - c["adaptive"]["plain"].get("steps", np.nan),
        "policy steps vs frames before success (mean VIG, x1000)":
            lambda c: -1000 * (c["stats"]["policy_all"].get("shift_vs_pre_success") or np.nan),
        "policy steps vs all frames (mean VIG, x1000)": lambda c: -1000 * (c["stats"]["policy_all"].get("shift_vs_all") or np.nan),
    }
    for name, f in rules.items():
        values = [(c["threshold_pct"], float(f(c))) for c in cols]
        crossings = []
        for (p0, v0), (p1, v1) in zip(values, values[1:]):
            if np.isfinite(v0) and np.isfinite(v1) and (v0 > 0) != (v1 > 0):
                crossings.append(p0 + (p1 - p0) * v0 / (v0 - v1))
        out["rules"][name] = dict(values={f"p{p:g}": v for p, v in values}, crossings=crossings)
    return _round(out)


# -------------------------------------------------------------------------------------------------------- Markdown
def _f(x, digits: int = 1, pct: bool = False) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.{digits}f} %" if pct else f"{x:.{digits}f}"


def _p(x) -> str:
    return "–" if x is None else f"{x:.2g}"


def _dump(data: dict) -> str:
    """The JSON block: one top-level key per line, each collection on its own line."""
    parts = []
    for k, v in data.items():
        if k == "collections":
            inner = ",\n".join(json.dumps(c, separators=(",", ":")) for c in v)
            parts.append(f'"collections":[\n{inner}\n]')
        else:
            parts.append(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}")
    return "{\n" + ",\n".join(parts) + "\n}"


def markdown(data: dict) -> str:
    a = analysis(data)
    cfg = data["run"]["config"]
    d = cfg["dagger"]
    ref = data["reference"]
    cols = sorted(data["collections"], key=lambda c: c["threshold_pct"])
    pol = data["retraining"]["policies"]
    seed = data["run"]["moe"]["seed"]
    rt = cfg["retrain"]["seeds"][0]
    moe = pol.get(f"moe/sd{seed}")
    L = [f"# Threshold sweep report: {cfg['name']}", "",
         f"Written {data['created']} by `vig-dagger sweep-report` (format {data['version']}). Redraw the figures from this "
         "file alone: `python -m vig_dagger.sweep_report plot <this file> --out <dir>`. Every number below is also in "
         "the JSON block at the end.", "",
         "## Run", "",
         f"- MoE: {data['run']['moe']['source']}, factors {cfg['moe']['factors']}, "
         f"{(data['retraining']['trainings'].get(f'moe/sd{seed}', {}).get('info') or {}).get('training_demos', '?')} training "
         f"demos (validation left out); this config's base: {cfg['demos']['num_demos']} planner demos ({cfg['demos']['motion']}); "
         f"checkpoint sha256 `{(data['run']['moe']['checkpoint_sha256'] or '–')[:16]}…`",
         f"- DAgger: {d['num_demos']} successful episodes per threshold, expert `{d['expert']}`, take over after "
         f"{d['take_count']} scores above, hand back after {d['return_count']} below"
         + (", random move at every take-over" if "random_move" in d["variants"] else "")
         + f", scenes from {d['seed_start']}",
         f"- Dense policies: {cfg['demos']['num_demos']} planner demos + each collection's episodes"
         + (" (random-move steps cut)" if cfg["retrain"]["cut_random_moves"] else "")
         + f", {cfg['training']['epochs']} epochs; evaluation on {cfg['evaluation']['num_episodes']} scenes from "
         f"{cfg['evaluation']['seed_start']}",
         f"- Code: {data['run']['code'].get('branch')} {str(data['run']['code'].get('commit'))[:10]}"
         + (f" (modified: {', '.join(data['run']['code']['modified_files'])})" if data["run"]["code"].get("modified_files") else ""),
         f"- GPUs: {'; '.join(data['run']['gpus']) or '–'}"]
    if data.get("merged_from"):  # a report merged from several machines' reports
        L.append(f"- Merged from the reports of {len(data['merged_from'])} machines (`python -m vig_dagger.sweep_report merge`):")
    for m in data.get("merged_from", []):
        L.append(f"  - from `{m['report']}` (written {m['created']}, code {(m['code'].get('commit') or '–')[:10]}): collections "
                 f"{', '.join(Collection.parse(c).threshold for c in m['collections']) or '–'}; evaluations "
                 f"{', '.join(m['evaluations']) or '–'}; its thresholds differ from the first report's by at most "
                 f"{100 * m['thresholds_max_rel_diff']:.3g} %")
    L += [f"  - note: {n}" for n in data.get("merge_notes", [])]
    L.append("")
    # ---- results at a glance
    L += ["## Results", "",
          "| threshold | VIG | attempts | success / attempt | expert share of the data | dense success | 95 % interval | vs next lower threshold |",
          "|---|---|---|---|---|---|---|---|"]
    settings = {s["threshold"]: s for s in data["collections_figure"]["settings"]}
    crow = {r["threshold"]: r for r in data["collections_figure"].get("rows", [])}
    prev = None
    for c in cols:
        s = settings.get(c["threshold"], {})
        dn = a["dense"].get(c["threshold"])
        vs = a["paired"].get(f"{c['threshold']} vs {prev}") if prev else None
        dense = (f"{dn['successes']}/{dn['episodes']} = {100 * dn['rate']:.1f} %" if dn else "not evaluated")
        interval = (f"{100 * dn['low']:.1f}–{100 * dn['high']:.1f} %" if dn else "–")
        versus = (f"+{vs['only_first']} / -{vs['only_second']}, p {_p(vs['p'])}" if vs else "–")
        L.append(f"| {c['threshold']} | {c['threshold_vig']:.5f} | {c['attempts']} | {_f(c['successes'] / c['attempts'], 1, True)} "
                 f"| {_f((s.get('data_expert') or {}).get('mean'), 1, True)} | {dense} | {interval} | {versus} |")
        if dn:
            prev = c["threshold"]
    if moe:
        L += ["", f"MoE alone (full resolution): {moe['successes']}/{moe['episodes']}."]
    for k, v in pol.items():
        if k.startswith("extra/"):
            name = k[len("extra/"):]
            label = (cfg.get("extra_policies", {}).get(name) or {}).get("label", name)
            L.append(f"{label} (`{name}`, evaluated here): {v['successes']}/{v['episodes']}.")
    if a["best"]:
        b = a["best"]
        L += ["", "**Best threshold.** Highest dense success: " + ", ".join(b["raw_best"])
              + f". Peak of a quadratic fit over the thresholds: p{_f(b['quadratic']['vertex'])}"
              + ("" if b["quadratic"]["concave"] else " (the fit is not concave: no inner peak)")
              + f"; resampling the scenes {b['resampled_scenes']} times: median p{_f(b['quadratic_peak_median'])}, "
              f"80 % range p{_f(b['quadratic_peak_80'][0])}–p{_f(b['quadratic_peak_80'][1])}. How often each threshold "
              "is the best when the scenes are resampled: "
              + ", ".join(f"{k} {100 * v:.0f} %" for k, v in b["raw_best_share"].items()) + "."]
    paired = [f"{k}: +{v['only_first']} / -{v['only_second']}, p {_p(v['p'])}" for k, v in a["paired"].items() if v]
    if paired:
        L += ["", "Paired sign tests (scenes only the first solved / only the second; exact, uncorrected): "
              + "; ".join(paired) + "."]
    # ---- the rules
    L += ["", "## What the threshold rules say", "",
          "Positive: the rule would raise the threshold from there; it settles where the sign changes. Adaptive: "
          f"q(M) - mean q(m), q = percentile among the reference frames, M = mean VIG of all reference frames "
          f"(p{_f(ref['mean_all_pct'])}), m = an attempt's mean VIG over its counted policy steps (the scores that "
          "triggered a take-over left out); failed-attempt rule: a failed attempt's last policy stretch counts at the "
          "threshold. Policy steps vs frames: mean VIG of the reference frames minus that of all policy steps (x1000).", "",
          "| rule | " + " | ".join(c["threshold"] for c in cols) + " | settles at |",
          "|---|" + "---|" * len(cols) + "---|"]
    for name, r in a["rules"].items():
        L.append(f"| {name} | " + " | ".join(_f(v, 2) for v in r["values"].values()) + " | "
                 + (", ".join(f"p{x:.1f}" for x in r["crossings"]) or "no change of sign") + " |")
    # ---- the reference
    pa, pp = ref["percentiles_all"], ref["percentiles_pre_success"]
    L += ["", "## The MoE's reference frames", "",
          f"{ref['frames']} frames (every {ref['frame_stride']}th of the first {ref['first_demos']} planner demos; "
          f"{ref['demo_steps_mean']:.0f} steps per demo, {ref['holding_steps_mean']:.0f} of them after success, so "
          f"{100 * ref['holding_frame_share']:.0f} % of the frames). Mean VIG {ref['mean_all']:.5f} = "
          f"p{ref['mean_all_pct']:.1f}; before success {ref['mean_pre_success']:.5f} = p{ref['mean_pre_success_pct']:.1f} "
          "of all frames.", "",
          "| percentile | " + " | ".join(f"p{q}" for q in (50, 60, 65, 70, 75, 80, 85, 90, 95, 99)) + " |",
          "|---|" + "---|" * 10,
          "| all frames | " + " | ".join(f"{pa[q]:.5f}" for q in (50, 60, 65, 70, 75, 80, 85, 90, 95, 99)) + " |",
          "| before success | " + " | ".join(f"{pp[q]:.5f}" for q in (50, 60, 65, 70, 75, 80, 85, 90, 95, 99)) + " |"]
    # ---- collections in detail
    L += ["", "## Collections", "",
          "Expert / random-move shares pooled over steps. Policy steps: as the rules count them. Percentiles among the "
          "reference frames.", "",
          "| threshold | successes / attempts | data: expert + move | all: expert + move | failed: expert + move "
          "| take-overs per attempt (median) | policy-step mean | expert-step mean | policy steps above the threshold "
          "| expert steps above | failure reasons |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cols:
        r = crow.get(c["threshold"], {})
        sp, se = c["stats"]["policy_counted"], c["stats"]["expert"]
        reasons = r.get("reasons") or {}
        L.append(f"| {c['threshold']} | {c['successes']}/{c['attempts']} | {_f(r.get('data_expert'), 0, True)} + "
                 f"{_f(r.get('data_move'), 0, True)} | {_f(r.get('all_expert'), 0, True)} + {_f(r.get('all_move'), 0, True)} "
                 f"| {_f(r.get('failed_expert'), 0, True)} + {_f(r.get('failed_move'), 0, True)} "
                 f"| {_f(r.get('takeovers_median'), 0)} | p{_f(sp.get('mean_pct'))} | p{_f(se.get('mean_pct'))} "
                 f"| {_f(sp.get('above_threshold'), 1, True)} | {_f(se.get('above_threshold'), 1, True)} | "
                 + (", ".join(f"{k} {v}" for k, v in reasons.items()) or "–") + " |")
    # ---- dense policies in detail
    L += ["", "## Dense policies", "",
          "| policy | success | median steps of successes | training demos | updates | final / best validation loss "
          "| failures by stage |", "|---|---|---|---|---|---|---|"]
    tr = data["retraining"]["trainings"]
    for label, p in pol.items():
        t = tr.get(label, {})
        L.append(f"| `{label}` | {p['successes']}/{p['episodes']} | {_f(p['success_steps_median'], 0)} "
                 f"| {(t.get('info') or {}).get('training_demos', '–')} | {t.get('updates', '–')} "
                 f"| {_f(t.get('final_val_loss'), 4)} / {_f(t.get('best_val_loss'), 4)} | "
                 + (", ".join(f"{k} {v}" for k, v in p["failed_at"].items()) or "–") + " |")
    L += ["", "One character per evaluation scene (1 solved, 0 not, - not evaluated), scenes in order:", "", "```"]
    for label, p in pol.items():
        L.append(f"{label:<45s} {p['scenes']}")
    L += ["```", "", "## Data", "",
          "Everything above and the figures' data. Read it with `vig_dagger.sweep_report.read(<this file>)`.", "", MARK,
          "```json", _dump(data), "```", ""]
    return "\n".join(L)


def read(path: str | Path) -> dict:
    """The JSON block of a report."""
    return read_text_data(Path(path).read_text(), str(path))


def read_text_data(text: str, name: str = "the text") -> dict:
    """The JSON block of a report's text."""
    at = text.find(MARK)
    if at < 0:
        raise SystemExit(f"{name} has no sweep-report data block")
    start = text.index("```json", at) + len("```json")
    end = text.index("\n```", start)
    return json.loads(text[start:end])


def write(cfg: dict, figures: bool = True) -> Path:
    data = build(cfg)
    results = Paths.of(cfg).results()
    results.mkdir(parents=True, exist_ok=True)
    path = results / "SWEEP_REPORT.md"
    path.write_text(markdown(data))
    print(f"[sweep-report] {path} ({path.stat().st_size / 1000:.0f} kB)", flush=True)
    if figures:  # drawn from the file just written, as anyone with only the file would draw them
        made = plot(read(path), results / "figures" / "from_report")
        print(f"[sweep-report] figures from the report: {', '.join(str(m) for m in made)}", flush=True)
    return path


# -------------------------------------------------------------------------------------------------------- figures
def plot(data: dict, out: Path) -> list[Path]:
    from vig_dagger import evaluation, summary

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = data["run"]["config"]
    made = []
    settings = data["collections_figure"]["settings"]
    if settings:
        summary.figure(cfg, settings, data["collections_figure"].get("references", []), out / "collections.png")
        made.append(out / "collections.png")
    fig = data["retraining"]["figure"]
    if fig.get("retrained") or fig.get("moes"):
        evaluation.figure(cfg, fig, out / "retraining.png")
        made.append(out / "retraining.png")
    if data["collections"]:
        figure_distributions(data, out / "distributions.png")
        made.append(out / "distributions.png")
    return made


def figure_distributions(data: dict, path: Path) -> None:
    """One panel per threshold: VIG histograms (area 1) of the MoE's reference frames, the collection's counted policy
    steps, policy + expert steps and expert steps alone, with the reference mean, the policy-step mean and the
    threshold."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vig_dagger.summary import GRID, INK, MUTED

    cols = sorted(data["collections"], key=lambda c: c["threshold_pct"])
    ref = data["reference"]
    n = len(cols)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    width = 4.4 * ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, 1.6 + 4.1 * nrow), dpi=130, squeeze=False)
    top = 1 - 1.45 / (1.6 + 4.1 * nrow)
    fig.subplots_adjust(left=0.6 / width, right=0.985, top=top, bottom=0.5 / (1.6 + 4.1 * nrow), wspace=0.2, hspace=0.42)
    fig.suptitle(f"{data['run']['config']['name']}: VIG of the MoE's reference frames vs the DAgger collections", x=0.6 / width,
                 ha="left", fontsize=13, color=INK)
    note = ("Histograms scaled to area 1 (values past the right edge in the last bin). Grey: the MoE's VIG on its "
            f"reference frames (every {ref['frame_stride']}th frame of the first {ref['first_demos']} planner demos); "
            "dotted grey: the reference frames before each demo's success (without the holding steps after it). "
            "Thick: the collection's policy steps, every attempt, as the threshold rules count them (the scores that "
            "triggered a take-over left out). Thin: the same plus the scores after expert steps (random-move steps are "
            "not scored). Dash-dot: the scores after expert steps alone. Lines: the reference mean, the policy steps' "
            "mean and the threshold.")
    fig.text(0.6 / width, top + 0.9 / (1.6 + 4.1 * nrow), textwrap.fill(note, int(15 * width)), fontsize=7.8,
             color=MUTED, va="top")
    cmap = plt.get_cmap("viridis")
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        c = cols[k]
        f = c["figure"]
        color = cmap(0.1 + 0.75 * k / max(n - 1, 1))
        bins = np.linspace(0.0, f["right"], f["bins"] + 1)
        w = bins[1] - bins[0]
        dens = {key: np.asarray(v, float) / max(sum(v), 1) / w for key, v in f["counts"].items()}
        cnt = {key: sum(v) for key, v in f["counts"].items()}
        ax.stairs(dens["reference"], bins, color="#9ca3af", fill=True, alpha=0.45, label=f"reference frames ({cnt['reference']:,})")
        if "reference_pre_success" in dens:
            ax.stairs(dens["reference_pre_success"], bins, color="#6b7280", lw=0.9, ls=(0, (1, 1)),
                      label=f"reference before success ({cnt['reference_pre_success']:,})")
        ax.stairs(dens["policy"], bins, color=color, lw=1.9, label=f"policy steps ({cnt['policy']:,})")
        ax.stairs(dens["both"], bins, color=color, lw=0.9, alpha=0.75, label=f"policy + expert steps ({cnt['both']:,})")
        if cnt["expert"]:
            ax.stairs(dens["expert"], bins, color=color, lw=1.1, ls=(0, (3, 1.5, 1, 1.5)),
                      label=f"expert steps ({cnt['expert']:,}; mean p{f['expert_mean_pct']:.1f})")
        lines = [(ref["mean_all"], "-", INK, f"reference mean = p{ref['mean_all_pct']:.1f}")]
        if f["policy_mean"] is not None:
            lines.append((f["policy_mean"], (0, (4, 2)), color, f"policy-step mean = p{f['policy_mean_pct']:.1f}"))
        lines.append((c["threshold_vig"], (0, (1, 1.5)), INK, f"threshold = {c['threshold']}"))
        ymax = ax.get_ylim()[1]
        for (x, ls, col, label), y in zip(lines, (0.97, 0.88, 0.79)):
            x = min(x, f["right"])
            ax.axvline(x, color=col, lw=1.2, ls=ls)
            left = x > 0.6 * f["right"]
            ax.text(x + (-1 if left else 1) * f["right"] * 0.01, ymax * y, label, fontsize=7.3, color=INK, va="top",
                    ha="right" if left else "left", bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
        share = lambda x: "–" if x is None else f"{100 * x:.0f} %"  # noqa: E731
        ax.text(0.98, 0.6, f"overlap with the reference:\npolicy {share(f['overlap'])}, policy + expert "
                f"{share(f['overlap_both'])}\nabove the threshold:\npolicy {share(f['policy_above'])}, expert "
                f"{share(f['expert_above'])}", transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
        ax.set_title(f"{c['threshold']} ({c['threshold_vig']:.4f}) · {c['successes']} successes in {c['attempts']} attempts",
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xlim(0, f["right"])
        ax.set_xlabel("VIG", color=MUTED, fontsize=9)
        ax.legend(fontsize=6.8, frameon=True, framealpha=0.9, edgecolor="none", loc=(0.36, 0.17))
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        if k % ncol == 0:
            ax.set_ylabel("density", color=MUTED, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- reference for a download
def merge_reference(cfg: dict, seed: int) -> Path:
    """Merge ``workers/score_demos.py`` parts into ``demo_scores.npz`` (thresholds.json stays as downloaded)."""
    from vig_dagger.workers.score_demos import part_path

    paths = Paths.of(cfg)
    chunks = int(cfg["thresholds"]["chunks"])
    parts = [np.load(part_path(cfg, seed, c)) for c in range(chunks)]
    parts = [p for p in parts if len(p["vig"])]
    keys = [k for k in ("vig", "vig_gap", "flow", "demo", "frame") if k in parts[0].files]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    np.savez(paths.demo_scores(seed), **merged, factors=parts[0]["factors"])
    print(f"[sweep-report] {paths.demo_scores(seed)}: {len(merged['vig'])} frames", flush=True)
    return paths.demo_scores(seed)


# ------------------------------------------------------------------------------------- reports of several machines
def merge(reports: list[Path]) -> dict:
    """One report from the reports of several machines that each ran some of the experiment's tasks (assigned by hand
    with `vig-dagger run --tasks`, no board): their collections, trainings and evaluations together, and the
    comparisons between policies recomputed from each scene's result. The reports must share the experiment settings
    and the MoE checkpoint. Every machine scored the demos itself (task 1): the first report's reference is used, and
    each report's thresholds are compared with it (``merged_from``)."""
    import copy

    from vig_dagger import evaluation

    datas = [read(r) for r in reports]
    base = datas[0]
    for path, d in zip(reports[1:], datas[1:]):
        if d["run"]["config"] != base["run"]["config"]:
            raise SystemExit(f"{path} comes from other experiment settings than {reports[0]}")
        if d["run"]["moe"]["checkpoint_sha256"] != base["run"]["moe"]["checkpoint_sha256"]:
            raise SystemExit(f"{path} used another MoE checkpoint than {reports[0]}")
    cfg = base["run"]["config"]
    order = {c.id: k for k, c in enumerate(collections(cfg))}
    base_thr = base["run"]["moe"]["thresholds"]
    chosen: dict[str, tuple[int, dict]] = {}  # collection id -> (report index, its data)
    policies: dict[str, dict] = {}
    trainings: dict[str, dict] = {}
    datasets: dict[str, dict] = {}
    merged_from, notes = [], []
    for i, (path, d) in enumerate(zip(reports, datas)):
        for c in d["collections"]:
            if c["id"] in chosen:
                j, other = chosen[c["id"]]
                notes.append(f"{c['id']} is in {Path(reports[j]).name} and {Path(path).name}; the one with more attempts "
                             "is used")
                if c["attempts"] <= other["attempts"]:
                    continue
            chosen[c["id"]] = (i, c)
        for label, pol in d["retraining"]["policies"].items():
            if label in policies:
                if pol["episodes"] > policies[label]["episodes"]:
                    policies[label] = pol
                elif pol["episodes"] == policies[label]["episodes"] and pol["scenes"] != policies[label]["scenes"]:
                    notes.append(f"{label} was evaluated in several reports with different results; the first is used")
                continue
            policies[label] = pol
        for label, t in d["retraining"]["trainings"].items():
            if label not in trainings or (not trainings[label].get("info") and t.get("info")):
                trainings[label] = t
        for name, ds in d["retraining"]["datasets"].items():
            datasets.setdefault(name, ds)
        thr = d["run"]["moe"]["thresholds"]
        merged_from.append(dict(
            report=Path(path).name, created=d["created"], code=d["run"]["code"], gpus=d["run"]["gpus"],
            collections=[c["id"] for c in d["collections"]], evaluations=sorted(d["retraining"]["policies"]),
            reference_frames=d["reference"]["frames"],
            thresholds_max_rel_diff=max((abs(thr[k] - v) / abs(v) for k, v in base_thr.items() if k in thr and v),
                                        default=0.0)))
    ids = sorted(chosen, key=lambda c: order.get(c, len(order)))
    source = {cid: chosen[cid][0] for cid in ids}
    rows, settings = [], []
    for cid in ids:  # a collection's figure rows come from the report its data comes from
        d = datas[source[cid]]
        col = Collection.parse(cid)
        rows += [r for r in d["collections_figure"]["rows"] if r["collection"] == cid]
        settings += [s for s in d["collections_figure"]["settings"]
                     if (s["scorer"], s["variant"], s["threshold"]) == (col.scorer, col.variant, col.threshold)]
    first, n_eval = int(cfg["evaluation"]["seed_start"]), int(cfg["evaluation"]["num_episodes"])
    complete = {label: [dict(seed=first + k, success=ch == "1") for k, ch in enumerate(pol["scenes"])]
                for label, pol in policies.items() if pol["episodes"] >= n_eval and "-" not in pol["scenes"]}
    ev = evaluation.comparisons(dict(cfg, evaluation=dict(cfg["evaluation"], extra_policies=cfg.get("extra_policies") or {})),
                                complete)
    fig = {k: ev.get(k) for k in ("episodes_per_policy", "scenes", "moes", "moe_counts", "settings", "extra")}
    fig["retrained"] = ev["retrained"]
    fig["references"] = base["retraining"]["figure"].get("references", [])
    out = copy.deepcopy(base)
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    out["run"]["gpus"] = sorted({g for d in datas for g in d["run"]["gpus"]})
    out["collections"] = [chosen[cid][1] for cid in ids]
    out["collections_figure"] = dict(settings=settings, references=base["collections_figure"].get("references", []),
                                     rows=rows)
    out["retraining"] = dict(policies=dict(sorted(policies.items())), trainings=trainings, datasets=datasets, figure=fig)
    out["merged_from"] = merged_from
    out["merge_notes"] = notes
    return _round(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["write", "plot", "merge-reference", "analysis", "merge"])
    p.add_argument("report", nargs="*", type=Path, help="plot / analysis: the report file; merge: the reports of every "
                   "machine (the first one's reference is used)")
    p.add_argument("--config", type=Path, help="write / merge-reference: the experiment config (or its config.json)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="plot: the folder for the figures (default: next to the report); merge: "
                   "the merged report (default: SWEEP_REPORT_merged.md next to the first report)")
    args = p.parse_args()
    if args.command in ("write", "merge-reference"):
        from vig_dagger import config as config_module

        cfg = config_module.load(args.config)
        if args.command == "write":
            write(cfg)
        else:
            merge_reference(cfg, args.seed)
    elif args.command == "merge":
        if len(args.report) < 2:
            p.error("merge takes the reports of at least two machines")
        data = merge(args.report)
        out = args.out or args.report[0].parent / "SWEEP_REPORT_merged.md"
        out.write_text(markdown(data))
        print(f"[sweep-report] {out}: {len(data['collections'])} collections, {len(data['retraining']['policies'])} "
              f"evaluated policies from {len(args.report)} reports" + "".join(f"\n  note: {n}" for n in data["merge_notes"]))
        made = plot(read(out), out.parent / f"figures_{out.stem}")
        print("\n".join(str(m) for m in made))
    else:
        if len(args.report) != 1:
            p.error(f"{args.command} takes one report")
        if args.command == "plot":
            made = plot(read(args.report[0]), args.out or args.report[0].parent / "figures_from_report")
            print("\n".join(str(m) for m in made))
        else:
            print(json.dumps(analysis(read(args.report[0])), indent=1))


if __name__ == "__main__":
    main()
