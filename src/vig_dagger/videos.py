"""Videos of DAgger attempts with the VIG score in sync (``<work_dir>/videos/<collection>/``).

For each chosen attempt one mp4 at 20 fps (real time; the simulation runs at 20 Hz):

  top     the cameras at the current step
  bottom  the score against the step: scores while the policy drives (solid) and while the expert drives (dashed; they
          decide the hand-back), the threshold the attempt ran with (iteration mode: its own adaptive threshold, with
          its percentile of the round's reference), expert control shaded grey, random moves shaded orange, gripper
          closes marked, a cursor, and a status line (step, score, who drives, plug-to-goal distance, plug held)

Which attempts: ``videos.per_kind`` successful ones (``trajectories/``) and as many failed ones
(``trajectories_failed/``, saved when ``dagger.save_failed_trajectories`` is true), spread evenly over the attempts in
the order they finished, the first and the last included. Files: ``success/`` and ``failed/``, named
``attempt<k>_seed<s>.mp4`` (k: the attempt's place in that order); ``index.json`` lists them.

    vig-dagger videos -c <config> [--collection <id>]            (the queue runs it after each collection's analysis)
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # the workers may still hold their files open
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from vig_dagger.config import Collection, Paths  # noqa: E402
from vig_dagger.recording import EXPERT, MOVE, POLICY  # noqa: E402

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
SCORE_COLOR = "#eb6834"
EXPERT_SHADE, MOVE_SHADE, CLOSE_COLOR = "#d1d5db", "#f59e0b", "#7c3aed"
CAMERAS = ("third_person_camera", "side_camera", "hand_camera")
FPS = 20


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2].tolist(), d[1::2].tolist()))


def _line(idx: np.ndarray, values: np.ndarray, gap) -> tuple[np.ndarray, np.ndarray]:
    """x, y through (idx, values[idx]) with a break wherever gap(previous index, index) is true."""
    xs, ys = [], []
    for k, t in enumerate(idx):
        if k and gap(idx[k - 1], t):
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(float(t))
        ys.append(float(values[t]))
    return np.asarray(xs, float), np.asarray(ys, float)


def _writer(path: Path, width: int, height: int) -> subprocess.Popen:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt",
           "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def render(job: dict) -> str:
    """Write one attempt's video (``job``: h5, key, out, title, threshold, return_threshold, threshold_pct,
    scorer, take_count, return_count)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with h5py.File(job["h5"], "r") as f:
        g = f[job["key"]]
        cams = [c for c in CAMERAS if f"obs/sensor_data/{c}/rgb" in g]
        frames = {c: g[f"obs/sensor_data/{c}/rgb"][:] for c in cams}
        score = g["dagger/score"][:].astype(float)
        ctrl = g["dagger/controller"][:].astype(int)
        dist = g["task_state/dist"][:].astype(float) if "task_state/dist" in g else None
        held = g["task_state/held"][:].astype(bool) if "task_state/held" in g else None
        grip = g["actions"][:, -1]
    thr, rthr, scorer = float(job["threshold"]), float(job["return_threshold"]), job["scorer"]
    n, steps = len(frames[cams[0]]), len(ctrl)  # observations (steps + 1), steps
    came_from = np.r_[POLICY, ctrl][:n]  # who drove into observation t (t = 0: the reset)
    scored = np.flatnonzero(np.isfinite(score))
    pol = scored[came_from[scored] != EXPERT]  # scored while the policy drives (incl. the take-over triggers)
    exp = scored[came_from[scored] == EXPERT]  # scored after expert steps
    px, py = _line(pol, score, lambda i, j: bool((ctrl[i:j] != POLICY).any()))
    ex, ey = _line(exp, score, lambda i, j: j - i > 1)
    expert_runs, move_runs = _runs(ctrl == EXPERT), _runs(ctrl == MOVE)
    closes = (np.flatnonzero((grip[1:] < 0) & (grip[:-1] >= 0)) + 1).tolist()
    finite = score[np.isfinite(score)]
    ymax = max(float(finite.max()) if len(finite) else 0.0, thr) * 1.2 or 1.0

    fig = plt.figure(figsize=(max(10.0, 4.6 * len(cams)), 7.4), dpi=100)
    gs = fig.add_gridspec(2, len(cams), height_ratios=[1.15, 1.0], hspace=0.22, wspace=0.05, left=0.07, right=0.98,
                          top=0.91, bottom=0.17)
    images = []
    for j, c in enumerate(cams):
        a = fig.add_subplot(gs[0, j])
        images.append(a.imshow(frames[c][0]))
        a.set_xlabel(c.replace("_", " "), fontsize=10, color=INK)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    ax = fig.add_subplot(gs[1, :])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for a, b in expert_runs:
        ax.axvspan(a, b, color=EXPERT_SHADE, alpha=0.55, lw=0)
    for a, b in move_runs:
        ax.axvspan(a, b, color=MOVE_SHADE, alpha=0.35, lw=0)
    for c in closes:
        ax.axvline(c, color=CLOSE_COLOR, lw=1.0, ls=":")
    ax.plot(px, py, color=SCORE_COLOR, lw=1.2, alpha=0.25)
    (live,) = ax.plot([], [], color=SCORE_COLOR, lw=1.8, label=f"{scorer} while the policy drives")
    ax.plot(ex, ey, color=SCORE_COLOR, lw=1.0, ls="--", alpha=0.25)
    (live_ex,) = ax.plot([], [], color=SCORE_COLOR, lw=1.4, ls="--", label=f"{scorer} while the expert drives (hand-back check)")
    pct = f" = p{job['threshold_pct']:.1f} of the reference" if job.get("threshold_pct") is not None else ""
    if rthr == thr:
        ax.axhline(thr, color=INK, lw=1.2, ls="--", label=f"threshold {thr:.4g}{pct} (take-over after {job['take_count']} "
                   f"above, hand-back after {job['return_count']} below)")
    else:
        ax.axhline(thr, color=INK, lw=1.2, ls="--", label=f"take-over {thr:.4g} ({job['take_count']} scores above)")
        ax.axhline(rthr, color="#2563eb", lw=1.2, ls="-.", label=f"hand-back {rthr:.4g} ({job['return_count']} in a row below)")
    cursor = ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("step", color=MUTED, fontsize=9)
    ax.set_ylabel(f"{scorer}", color=MUTED, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=EXPERT_SHADE, alpha=0.55))
    labels.append("expert in control")
    if move_runs:
        handles.append(Patch(color=MOVE_SHADE, alpha=0.35))
        labels.append("random move (not scored)")
    if closes:
        handles.append(plt.Line2D([], [], color=CLOSE_COLOR, lw=1.0, ls=":"))
        labels.append("gripper closes")
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.suptitle(job["title"], fontsize=11, color=INK, x=0.07, y=0.985, ha="left")
    status = fig.text(0.07, 0.935, "", fontsize=11, color=INK, family="monospace")

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    width, height = width - width % 2, height - height % 2  # yuv420p needs even sides
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    proc = _writer(tmp, width, height)
    last = np.nan
    try:
        for t in range(n):
            for im, c in zip(images, cams):
                im.set_data(frames[c][t])
            live.set_data(px[~(px > t)], py[~(px > t)])
            live_ex.set_data(ex[~(ex > t)], ey[~(ex > t)])
            cursor.set_xdata([t, t])
            if np.isfinite(score[t]):
                last = score[t]
            c = ctrl[t] if t < steps else ctrl[-1]
            d = f"   plug-goal {1000 * dist[t]:5.1f} mm" if dist is not None else ""
            h = ("   held" if held[t] else "   not held") if held is not None else ""
            if c == MOVE:
                status.set_text(f"step {t:3d}   {scorer}: not scored   control: RANDOM MOVE{d}{h}")
                status.set_color("#b45309")
            else:
                above = bool(np.isfinite(last) and last > thr)
                below = bool(np.isfinite(last) and last < rthr)
                mark = (" > threshold" if above else "            ") if c != EXPERT else (" < threshold" if below else "            ")
                status.set_text(f"step {t:3d}   {scorer} = {last:<8.4g}{mark}   control: "
                                f"{'EXPERT (motion planner)' if c == EXPERT else 'policy'}{d}{h}")
                status.set_color("#b91c1c" if above and c == POLICY else INK)
            fig.canvas.draw()
            proc.stdin.write(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:height, :width, :3]).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {out}: {err}")
        tmp.replace(out)
    finally:
        if proc.poll() is None:
            proc.kill()
        tmp.unlink(missing_ok=True)
        plt.close(fig)
    return str(out)


def _summaries(out: Path) -> list[dict]:
    """Every finished attempt's summary line, in the order the attempts finished."""
    rows = []
    for path in sorted((out / "episodes").glob("worker_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["finished_at"])


def _saved(out: Path) -> dict[int, tuple[Path, str]]:
    """Scene seed -> (trajectory file, traj_<n>) of every saved attempt, successful and failed."""
    found = {}
    for sub in ("trajectories", "trajectories_failed"):
        for js in sorted((out / sub).glob("*.json")):
            try:
                episodes = json.loads(js.read_text()).get("episodes", [])
            except (OSError, ValueError):
                continue
            for e in episodes:
                if "seed" in e:
                    found[int(e["seed"])] = (js.with_suffix(".h5"), f"traj_{e['episode_id']}")
    return found


def spread(n: int, k: int) -> list[int]:
    """k indices spread evenly over range(n), the first and the last included (all of them when k >= n)."""
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def choose(cfg: dict, collection: Collection, per_kind: int) -> tuple[list[dict], int]:
    """The attempts to render (as ``render`` jobs) and the collection's attempts so far."""
    out = Paths.of(cfg).dagger(collection)
    settings = json.loads((out / "settings.json").read_text())
    adaptive = bool(settings.get("adaptive"))
    rows = _summaries(out)
    saved = _saved(out)
    target = Paths.of(cfg).root / "videos" / collection.id
    jobs = []
    for kind, ok in (("success", True), ("failed", False)):
        candidates = [(k, r) for k, r in enumerate(rows, start=1) if bool(r["success"]) == ok and int(r["seed"]) in saved]
        for i in spread(len(candidates), per_kind):
            k, r = candidates[i]
            h5, key = saved[int(r["seed"])]
            thr = float(r.get("threshold", settings["take_threshold"])) if adaptive else float(settings["take_threshold"])
            rthr = thr if adaptive else float(settings["return_threshold"])
            moves = sum(not m.get("skipped") for m in r.get("random_moves") or [])
            title = (f"{collection.id} · attempt {k} of {len(rows)} · seed {r['seed']} — "
                     f"{'SUCCESS' if ok else 'FAILED'} ({r['reason']}; {r['steps']} steps, {r['takeovers']} take-over"
                     f"{'s' if r['takeovers'] != 1 else ''}, {moves} random move{'s' if moves != 1 else ''})")
            jobs.append(dict(h5=str(h5), key=key, out=str(target / kind / f"attempt{k:04d}_seed{r['seed']}.mp4"),
                             title=title, threshold=thr, return_threshold=rthr, threshold_pct=r.get("threshold_pct"),
                             scorer=collection.scorer, take_count=int(settings["take_count"]),
                             return_count=int(settings["return_count"]), seed=int(r["seed"]), attempt=k, success=ok,
                             reason=r["reason"]))
    return jobs, len(rows)


def index_path(cfg: dict, collection: Collection) -> Path:
    return Paths.of(cfg).root / "videos" / collection.id / "index.json"


def current(cfg: dict, collection: Collection) -> bool:
    """The collection's videos exist for every attempt it has now and the current ``videos.per_kind``."""
    from vig_dagger.workers.dagger import count_successes

    try:
        index = json.loads(index_path(cfg, collection).read_text())
    except (OSError, ValueError):
        return False
    return (index.get("per_kind") == int(cfg["videos"]["per_kind"])
            and index.get("attempts") == count_successes(Paths.of(cfg).dagger(collection))[1])


def render_collection(cfg: dict, collection: Collection, per_kind: int | None = None, workers: int | None = None) -> Path:
    """Render the chosen attempts of one collection (videos already there are kept); returns its video folder."""
    per_kind = int(cfg["videos"]["per_kind"] if per_kind is None else per_kind)
    workers = int(cfg["videos"]["workers"] if workers is None else workers)
    jobs, attempts = choose(cfg, collection, per_kind)
    todo = [j for j in jobs if not Path(j["out"]).is_file()]
    print(f"[videos] {collection.id}: {attempts} attempts; {len(jobs)} chosen "
          f"({sum(j['success'] for j in jobs)} successful, {sum(not j['success'] for j in jobs)} failed), "
          f"{len(todo)} to render", flush=True)
    failed = []
    if todo:
        with multiprocessing.get_context("spawn").Pool(max(1, min(workers, len(todo)))) as pool:
            results = [pool.apply_async(render, (j,)) for j in todo]
            for i, (j, res) in enumerate(zip(todo, results), start=1):
                try:
                    print(f"[videos] {i}/{len(todo)} {res.get()}", flush=True)
                except Exception as exc:  # an unreadable trajectory (e.g. still being written): skip it, say so
                    failed.append(j["out"])
                    print(f"[videos] {i}/{len(todo)} FAILED {j['out']}: {exc}", flush=True)
    done = [j for j in jobs if Path(j["out"]).is_file()]
    folder = index_path(cfg, collection).parent
    folder.mkdir(parents=True, exist_ok=True)
    keep = ("seed", "attempt", "success", "reason", "threshold", "threshold_pct")
    index = dict(collection=collection.id, attempts=attempts, per_kind=per_kind,
                 videos=[dict(file=str(Path(j["out"]).relative_to(folder)), **{k: j[k] for k in keep}) for j in done])
    if failed:
        raise SystemExit(f"[videos] {len(failed)} videos failed (see above); {len(done)} written to {folder}")
    index_path(cfg, collection).write_text(json.dumps(index, indent=1))
    print(f"[videos] {len(done)} videos in {folder}", flush=True)
    return folder
