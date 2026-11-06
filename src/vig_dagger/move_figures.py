"""Random moves seen from the third-person camera: for every MoE seed and threshold, the random_move collections of
VIG and flow loss side by side (like the earlier PlugCharger random-move figures).

For every random move in a collection's DAgger data (its first ``dagger.num_demos`` successful episodes): the outline
of the sphere the target was drawn from (``dagger.random_move.radius`` around the gripper tip at the take-over, drawn
as its true perspective silhouette), a dot at the centre, and an arrow to where the tip ended. Two figures per MoE
seed and threshold, in ``results/figures/moves/``:

  spheres_held_sd<s>_p<q>.png       colour = plug held or not at the take-over
  spheres_progress_sd<s>_p<q>.png   colour = when in the episode the take-over happened (start step / episode length)

Background (grayscale, lightened): the third-person camera's first frame of scene ``dagger.seed_start`` (or of the
collection's saved episode with the lowest seed). Input: ``results/collections/<collection>_moves.json`` and
``_background.png``, written by ``analysis.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vig_dagger.config import Collection, Paths

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP_COLORS = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABEL = {"vig": "VIG", "flow": "flow loss"}
DPI = 150


class Camera:
    """Pinhole projection of the third-person camera (OpenCV conventions, from the recorded sensor parameters)."""

    def __init__(self, extrinsic_cv, intrinsic_cv) -> None:
        e = np.asarray(extrinsic_cv, float)
        self.R, self.t, self.K = e[:, :3], e[:, 3], np.asarray(intrinsic_cv, float)
        self.C = -self.R.T @ self.t  # camera centre (world)

    def project(self, x) -> np.ndarray:
        c = np.atleast_2d(x) @ self.R.T + self.t
        p = c @ self.K.T
        return p[:, :2] / p[:, 2:3]

    def silhouette(self, centre, radius: float, n: int = 72) -> np.ndarray:
        """Image outline of a sphere: the projection of its limb circle (where view rays are tangent)."""
        v = np.asarray(centre, float) - self.C
        dd = np.linalg.norm(v)
        u = v / dd
        q = self.C + v * (1 - radius ** 2 / dd ** 2)
        rho = radius * np.sqrt(1 - radius ** 2 / dd ** 2)
        a = np.cross(u, [0, 0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(u, a)
        th = np.linspace(0, 2 * np.pi, n)
        return self.project(q + rho * (np.outer(np.cos(th), a) + np.outer(np.sin(th), b)))


def _background(arr: np.ndarray) -> np.ndarray:
    g = arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    return 0.5 * g + 0.5 * 255  # grayscale, lightened so the marks carry the colour


def _load(cfg: dict, collection: Collection):
    stem = Paths.of(cfg).collection_results(collection)
    path, bg = Path(f"{stem}_moves.json"), Path(f"{stem}_background.png")
    if not path.is_file():
        return None
    import matplotlib.image

    d = json.loads(path.read_text())
    image = matplotlib.image.imread(bg) if bg.is_file() else None
    if image is not None and image.dtype != np.uint8:
        image = (image * 255).round()
    return d, image


def figure(cfg: dict, seed: int, spec: str, mode: str, path: Path) -> bool:
    """One figure (mode "held" or "progress") of MoE ``seed`` at threshold ``spec``; False when neither scorer's
    random_move collection is analyzed yet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.lines import Line2D

    scorers = list(cfg["dagger"]["scorers"])
    data = {s: _load(cfg, Collection(seed, s, spec, "random_move")) for s in scorers}
    if not any(v is not None for v in data.values()):
        return False
    ramp = LinearSegmentedColormap.from_list("blue_ramp", RAMP_COLORS)
    radius = next(v[0]["radius"] for v in data.values() if v is not None)
    n_demos = int(cfg["dagger"]["num_demos"])
    rng = np.random.default_rng(0)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK}):
        fig, axes = plt.subplots(1, len(scorers), figsize=(7 * len(scorers), 8.1), dpi=DPI, facecolor=SURF, squeeze=False)
        fig.subplots_adjust(left=0.02, right=0.98 if mode == "held" else 0.90, top=0.83, bottom=0.08, wspace=0.05)
        backgrounds = set()
        for ax, scorer in zip(axes[0], scorers):
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(LABEL[scorer], loc="left", fontsize=13, fontweight="bold", pad=22)
            loaded = data[scorer]
            if loaded is None or loaded[0]["camera"] is None:
                ax.set_facecolor("#efeeea")
                ax.text(0.5, 0.5, "not analyzed yet" if loaded is None else "no successful episode yet", transform=ax.transAxes,
                        ha="center", va="center", color=MUTED)
                continue
            d, image = loaded
            cam = Camera(d["camera"]["extrinsic_cv"], d["camera"]["intrinsic_cv"])
            m = [x for x in d["moves"] if x["data"]]
            m = [m[i] for i in rng.permutation(len(m))]  # random draw order: no colour sits on top by construction
            n_eps = len({x["seed"] for x in m})
            size = image.shape[1] if image is not None else int(2 * cam.K[0, 2])
            if image is not None:
                ax.imshow(_background(image), cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
                backgrounds.add(d["background_seed"])
            if m:
                cols = [to_rgba(ORANGE if x["held"] else BLUE) if mode == "held" else ramp(x["progress"]) for x in m]
                ax.add_collection(LineCollection([cam.silhouette(x["start"], radius) for x in m],
                                                 colors=[(*c[:3], 0.16) for c in cols], linewidths=0.6, zorder=2))
                p0 = cam.project(np.array([x["start"] for x in m]))
                p1 = cam.project(np.array([x["end"] for x in m]))
                ax.quiver(p0[:, 0], p0[:, 1], p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1], color=[(*c[:3], 0.75) for c in cols],
                          angles="xy", scale_units="xy", scale=1, width=0.0016, headwidth=4, headlength=5,
                          headaxislength=4.5, zorder=3)
                ax.scatter(p0[:, 0], p0[:, 1], s=7, c=[(*c[:3], 0.9) for c in cols], linewidths=0, zorder=4)
            ax.set_xlim(0, size - 1); ax.set_ylim(size - 1, 0)
            ax.text(0, 1.012, f"{len(m):,} random moves in {n_eps} successful episodes"
                    + (f"  ·  plug held at {sum(x['held'] for x in m)}" if mode == "held" else ""),
                    transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
        fig.text(0.02, 0.955, f"Random end-effector moves at take-over, MoE seed {seed} at {spec}, from the third-person camera",
                 fontsize=14, fontweight="bold", color=INK)
        fig.text(0.02, 0.922, f"Circle: the {radius * 100:g} cm sphere the target was drawn from, centred on the gripper tip "
                 "at the take-over.  Dot: that centre.  Arrow: where the tip actually moved.", fontsize=10, color=INK2)
        scene = (f"first frame of scene {', '.join(f'{b:,}' for b in sorted(backgrounds))}" if backgrounds else "none")
        fig.text(0.02, 0.03, f"Random-move DAgger, each collection's DAgger data (its first {n_demos} successful episodes). "
                 f"Background: {scene} (grayscale). The socket's place changes between scenes.", fontsize=8.5, color=MUTED)
        if mode == "held":
            fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=5, lw=1.5, label="plug not held"),
                                Line2D([], [], color=ORANGE, marker="o", ms=5, lw=1.5, label="plug held")],
                       loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=2, frameon=False, fontsize=10)
        else:
            cax = fig.add_axes([0.915, 0.2, 0.012, 0.55])
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ramp, norm=plt.Normalize(0, 1)), cax=cax)
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1]); cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
            cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=9, colors=INK2)
            cb.set_label("take-over step / episode length", color=INK2, fontsize=9.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, facecolor=SURF)
        plt.close(fig)
    return True


def figures(cfg: dict, figures_dir: Path) -> list[tuple[int, str, list[str]]]:
    """Every figure that has data: (MoE seed, threshold, [figure paths relative to figures_dir])."""
    if "random_move" not in cfg["dagger"]["variants"]:
        return []
    made = []
    for seed in (int(s) for s in cfg["moe"]["seeds"]):
        for spec in cfg["dagger"]["thresholds"]:
            names = []
            for mode in ("held", "progress"):
                name = f"moves/spheres_{mode}_sd{seed}_{spec}.png"
                if figure(cfg, seed, spec, mode, figures_dir / name):
                    names.append(name)
            if names:
                made.append((seed, spec, names))
    return made
