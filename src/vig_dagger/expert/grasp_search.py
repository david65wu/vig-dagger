"""DAgger grasp candidates; no simulator dependency.

This is a conservative wrist-roll heuristic, not a collision or full-arm IK solver.
Execution must still verify that the requested pose was reached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class GraspYaw:
    yaw: float
    margin: float
    turn: float


def grasp_yaws(ideal_yaws, current, q7, limits, to_object, to_place, *,
               tolerance=math.radians(25), margin=0.3, excluded=()):
    """Search the allowed yaw intervals, including their interiors and endpoints.

Check wrist roll after in-place alignment, the swing to the object, and the
subsequent swing to placement. Prefer a little extra reserve over a tiny turn.
Unlike the legacy selector, never silently accept an infeasible candidate.
"""
    options = []
    for ideal in ideal_yaws:
        nearest = ideal + max(-tolerance, min(tolerance, wrap(current - ideal)))
        angles = [nearest] + [ideal + tolerance * i / 5 for i in range(-5, 6)]
        for yaw in angles:
            yaw = wrap(yaw)
            if any(abs(wrap(yaw - previous)) < math.radians(10) for previous in excluded):
                continue
            if any(abs(wrap(yaw - previous.yaw)) < 1e-6 for previous in options):
                continue
            turn = wrap(yaw - current)
            predicted = [q7 - turn, q7 + to_object - turn, q7 + to_object + to_place - turn]
            reserve = min(min(j - limits[0], limits[1] - j) for j in predicted)
            if reserve >= margin:
                options.append(GraspYaw(yaw, reserve, abs(turn)))
    return sorted(options, key=lambda x: (x.turn + 2 * max(0, 0.6 - x.margin), -x.margin))
