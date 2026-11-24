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


def step(p: float, total: int, n: int, mean: float | None, target_pct: float, reference: np.ndarray) -> tuple[float, int]:
    """One update: (p, N) after an attempt with n policy steps of mean VIG ``mean`` (no change without policy steps)."""
    if n <= 0 or mean is None:
        return float(p), int(total)
    total = int(total) + int(n)
    return float(np.clip(p + n / total * (target_pct - percentile_of(reference, mean)), 0.0, 100.0)), total


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

    def update(self, scores: list[float], used_p: float, **info) -> dict:
        """Apply one attempt's policy-step scores (trigger scores already left out); returns the log entry."""
        n = len(scores)
        mean = float(np.mean(scores)) if n else None
        with self._locked():
            state = self._read()
            p_next, total = step(state["p"], state["total_policy_steps"], n, mean, self.target_pct, self.reference)
            entry = dict(info, used_p=float(used_p), used_threshold=threshold_at(self.reference, used_p),
                         p_before=float(state["p"]), n=n, mean=mean,
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
