"""Who drives during a DAgger episode: the policy or the expert, decided from the query score with two thresholds.

- Policy driving: the expert takes over once the score is above ``take_threshold`` on ``take_count`` consecutive
  scored steps.
- Expert driving: the policy takes back control once the score is below ``return_threshold`` on ``return_count``
  consecutive scored steps.

Both counts default to 2. The counters restart at every switch, so a hand-over always needs fresh evidence.
``return_threshold`` is normally at most ``take_threshold`` (a band in between keeps whoever is driving).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TwoThresholdGate:
    take_threshold: float
    return_threshold: float
    take_count: int = 2
    return_count: int = 2

    expert_driving: bool = False
    count: int = 0  # consecutive scored steps meeting the current mode's switch condition

    def __post_init__(self) -> None:
        if self.take_count < 1 or self.return_count < 1:
            raise ValueError("take_count and return_count must be at least 1")

    def reset(self) -> None:
        self.expert_driving, self.count = False, 0

    def update(self, score: float) -> str | None:
        """Feed one score; returns "take" or "return" when control switches, else None."""
        if self.expert_driving:
            self.count = self.count + 1 if score < self.return_threshold else 0
            if self.count >= self.return_count:
                self.expert_driving, self.count = False, 0
                return "return"
        else:
            self.count = self.count + 1 if score > self.take_threshold else 0
            if self.count >= self.take_count:
                self.expert_driving, self.count = True, 0
                return "take"
        return None
