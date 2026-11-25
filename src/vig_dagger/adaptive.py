"""The adaptive threshold of iteration mode: one threshold for take-over and hand-back, moved after every attempt so
that the VIG of the steps the policy drives matches the VIG of the MoE's own training frames.

The threshold is kept as a percentile ``p`` of the reference (the MoE's VIG on every ``thresholds.frame_stride``-th
frame of all its training demos, ``demo_scores.npz``); the VIG threshold is that percentile of the reference. After
each attempt, successful or failed, with ``m`` the mean VIG of its policy steps and ``n`` their number:

    N <- N + n                                  (policy steps of every attempt so far)
    p <- clip(p + (n / N) * (q(M) - q(m)), 0, 100)

where ``M`` is the mean VIG of the reference and ``q(x)`` the percentile of ``x`` among the reference values. Nothing
is tuned: in percentile units moving the threshold by one percentile moves the policy steps' mean by about one
percentile (measured on the 1000-demo MoE's fixed-threshold collections), so the gain is 1, and the step is this
attempt's share of all policy steps so far (the first attempt moves p fully, later ones less and less; an attempt
with many policy steps counts more). The target ``q(M)`` is computed from each MoE's own reference: it depends on the
training data and the policy. With ``exclude_trigger`` the ``take_count`` scores that trigger each take-over leave
``m`` and ``n``: they are above the threshold by construction, and at low thresholds (the expert takes over within a
few steps) they are a large share of the few policy steps and make the mean look matched. An attempt without policy
steps leaves p unchanged.

With ``update_weight: attempts`` (``iterate.update_weight``) every attempt counts the same instead of by its policy
steps, and the start counts as one attempt: p <- p + (q(M) - q(m)) / (u + 2), u = the collection's earlier updates.
p is then the average of the start and each attempt's p + q(M) - q(m). With the default (n / N) a few attempts with
many policy steps -- e.g. failed attempts of a few-demo MoE at p100, where the expert cannot take over -- outweigh
everything after them, and at a low threshold, where an attempt has only a few policy steps, p hardly moves again
(round 1 of a 50-demo run fell from p100 to p35 in a few attempts and crept to p41 over 100).

With ``failed_last_stretch`` (``iterate.failed_last_stretch``) a FAILED attempt's last policy stretch -- its policy
steps after its last take-over, i.e. the policy drove into the failure without the gate firing; all its policy steps
if the expert never took over -- counts as if every one of those scores were at the threshold the attempt ran with:
steps the expert should have taken. Successful attempts, and a failed attempt's earlier policy steps, count as
without it; a failed attempt that ends with the expert driving has no last stretch. Without it a few-demo MoE, which
fails in states that look like its training frames, pushes p up after its failed attempts too (their policy steps
look familiar) until hardly anything is handed to the expert.

The collection's workers share the state (``threshold_state.json`` in the collection directory, updated under a file
lock): each reads p before an attempt and updates it after; ``threshold_log.jsonl`` gets one line per update.
"""

from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def percentile_of(reference: np.ndarray, value: float) -> float:
    """q(value): the share of reference values at or below it, in percent (``reference`` sorted)."""
    return float(100.0 * np.searchsorted(reference, value, side="right") / len(reference))


def threshold_at(reference: np.ndarray, p: float) -> float:
    """The VIG threshold at percentile p of the reference."""
    return float(np.percentile(reference, float(p)))


def counted_scores(scores, last_stretch_from: int | None, success: bool | None, threshold: float,
                   failed_last_stretch: bool) -> tuple[list[float], int]:
    """The scores the update uses and how many of them were set to the threshold: with ``failed_last_stretch``, in a
    failed attempt the scores from index ``last_stretch_from`` on (its last policy stretch) are set to ``threshold``."""
    scores = [float(x) for x in scores]
    if not failed_last_stretch or success is not False or last_stretch_from is None:
        return scores, 0
    k = max(0, min(int(last_stretch_from), len(scores)))
    return scores[:k] + [float(threshold)] * (len(scores) - k), len(scores) - k


def step(p: float, total: int, n: int, mean: float | None, target_pct: float, reference: np.ndarray,
         updates: int = 0, weight: str = "policy_steps") -> tuple[float, int]:
    """One update: (p, N) after an attempt with n policy steps of mean VIG ``mean`` (no change without policy steps).
    ``weight`` "policy_steps": the step is n / N (N = policy steps so far, this attempt's included); "attempts": every
    attempt counts the same and the start counts as one, so the step is 1 / (updates + 2) (``updates`` = earlier
    updates of this collection): p is the average of the start and each attempt's p + q(M) - q(m)."""
    if n <= 0 or mean is None:
        return float(p), int(total)
    total = int(total) + int(n)
    gain = n / total if weight == "policy_steps" else 1.0 / (int(updates) + 2)
    return float(np.clip(p + gain * (target_pct - percentile_of(reference, mean)), 0.0, 100.0)), total


def split_scores(score: np.ndarray, controller: np.ndarray, take_steps, take_count: int,
                 exclude_trigger: bool) -> tuple[np.ndarray, np.ndarray]:
    """One attempt's per-observation scores (``attempts/*.h5``: score, controller, take_steps) split into the
    policy-step scores the update uses (the worker's ``policy_scores``: every score computed while the policy drives,
    without the ``take_count`` that trigger each take-over when ``exclude_trigger``) and the scores computed after
    expert steps (the hand-back checks). Observation t is scored after step t-1, so who drove step t-1 says which."""
    from vig_dagger.recording import EXPERT, POLICY

    score = np.asarray(score, float)
    came_from = np.r_[POLICY, np.asarray(controller, int)][: len(score)]  # observation 0: the reset
    finite = np.isfinite(score)
    policy = np.flatnonzero(finite & (came_from != EXPERT))
    if exclude_trigger and len(take_steps):
        place = {int(t): i for i, t in enumerate(policy)}
        drop = set()
        for t in take_steps:
            i = place.get(int(t))
            if i is not None:
                drop.update(policy[max(0, i - int(take_count) + 1): i + 1].tolist())
        policy = np.array([t for t in policy if t not in drop], int)
    return score[policy], score[finite & (came_from == EXPERT)]


def load_reference(path: Path) -> np.ndarray:
    """The sorted reference VIG values (``demo_scores.npz``, key vig)."""
    return np.sort(np.load(path)["vig"].astype(float))


class SharedThreshold:
    """The adaptive threshold shared by a collection's workers (see the module docstring)."""

    def __init__(self, out: Path, settings: dict) -> None:
        a = settings["adaptive"]
        self.state_path, self.log_path = out / "threshold_state.json", out / "threshold_log.jsonl"
        self.lock_path = out / "threshold_state.lock"
        self.reference = load_reference(Path(a["reference"]))
        self.target_pct = float(a["target_pct"])
        self.start = float(a["start_percentile"])
        self.failed_last_stretch = bool(a.get("failed_last_stretch", False))
        self.weight = str(a.get("update_weight", "policy_steps"))

    @contextmanager
    def _locked(self):
        with open(self.lock_path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _read(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return dict(p=self.start, total_policy_steps=0, updates=0, attempts=0)

    def current(self) -> tuple[float, float]:
        """(percentile, VIG threshold) for the next attempt."""
        with self._locked():
            p = float(self._read()["p"])
        return p, threshold_at(self.reference, p)

    def update(self, scores: list[float], used_p: float, success: bool | None = None,
               last_stretch_from: int | None = None, **info) -> dict:
        """Apply one attempt's policy-step scores (trigger scores already left out; ``last_stretch_from``: index of the
        first score after the attempt's last take-over, 0 without one); returns the log entry."""
        raw = [float(x) for x in scores]
        scores, n_last = counted_scores(raw, last_stretch_from, success, threshold_at(self.reference, used_p),
                                        self.failed_last_stretch)
        n = len(scores)
        mean = float(np.mean(scores)) if n else None
        with self._locked():
            state = self._read()
            p_next, total = step(state["p"], state["total_policy_steps"], n, mean, self.target_pct, self.reference,
                                 updates=int(state["updates"]), weight=self.weight)
            entry = dict(info, success=success, used_p=float(used_p), used_threshold=threshold_at(self.reference, used_p),
                         p_before=float(state["p"]), n=n, mean=mean, n_last_stretch=n_last,
                         last_stretch_raw_mean=float(np.mean(raw[len(raw) - n_last:])) if n_last else None,
                         mean_pct=percentile_of(self.reference, mean) if n else None, target_pct=self.target_pct,
                         p_after=p_next, threshold_after=threshold_at(self.reference, p_next), total_policy_steps=total,
                         time=time.time())
            state.update(p=p_next, total_policy_steps=total, updates=int(state["updates"]) + bool(n),
                         attempts=int(state["attempts"]) + 1)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(self.state_path)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        return entry


def final_percentile(out: Path) -> float | None:
    """Where a collection's adaptive threshold ended (its state file), or None."""
    path = out / "threshold_state.json"
    return float(json.loads(path.read_text())["p"]) if path.is_file() else None
