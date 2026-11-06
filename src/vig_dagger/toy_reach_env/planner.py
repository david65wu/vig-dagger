"""Seeded Cartesian Bezier paths inside a narrow tube, tracked with relative IK.

The scene has no obstacles in the reaching workspace. A cubic curve's convex-hull
property bounds its perpendicular deviation by the control-point offsets.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TubePlan:
    control_points: np.ndarray
    point_a: np.ndarray
    point_b: np.ndarray
    seed: int

    @classmethod
    def sample(cls, point_a, point_b, radius, seed):
        a, b = np.asarray(point_a, dtype=np.float64), np.asarray(point_b, dtype=np.float64)
        direction = b - a
        length = np.linalg.norm(direction)
        if length < 1e-6 or not np.isfinite(length) or not np.isfinite(radius) or radius < 0:
            raise ValueError("Need distinct finite endpoints and a nonnegative finite radius")
        direction /= length
        helper = np.eye(3)[np.argmin(np.abs(direction))]
        u = np.cross(direction, helper)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi, 3)
        radii = radius * np.sqrt(rng.uniform(0, 1, 3))
        offsets = radii[:, None] * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
        points = a + np.linspace(0, 1, 4)[:, None] * (b - a)
        points[:3] += offsets
        return cls(points, a, b, seed)

    def evaluate(self, progress):
        t = np.asarray(progress).clip(0, 1)[..., None]
        c = self.control_points
        return (1 - t)**3 * c[0] + 3 * (1 - t)**2 * t * c[1] + 3 * (1 - t) * t**2 * c[2] + t**3 * c[3]

    def progress(self, position):
        d = self.point_b - self.point_a
        return float(np.clip(np.dot(np.asarray(position) - self.point_a, d) / np.dot(d, d), 0, 1))

    def target(self, position, lookahead=0.012):
        progress = self.progress(position)
        return self.evaluate(progress + lookahead / np.linalg.norm(self.point_b - self.point_a))
