"""DAgger-only charger alignment and bounded recovery.

The original planner is unchanged. This subclass searches valid grasp angles,
does not freeze away grasp orientation errors, and retries an approach at most
twice. It still uses the environment's Cartesian IK controller; it is not a
global collision-aware joint-trajectory planner.
"""
from __future__ import annotations

import math

import torch

from . import planner as P
from .grasp_search import grasp_yaws

REVISION = "dagger_recovery_v1"


class ApproachFailed(P.PlanFailed):
    """An uncompleted charger approach, before closing the gripper."""


class DaggerPlanner(P.OpenDrawerPlanner):
    def __init__(self, env, params, *, trace=None):
        super().__init__(env, params, expert=True, trace=trace)
        self.events = []
        self._tried_yaws = []

    def _event(self, kind, **details):
        self.events.append(dict(step=self.steps, kind=kind, **details))
        self._trace(f"{kind}: {details}")

    def _nearest_grasp_quat(self, ideal_yaws, az_delta, then_az_delta):
        options = grasp_yaws(ideal_yaws, self.hand_yaw(), self.q7(), self._q7_limits,
                             az_delta, then_az_delta, excluded=self._tried_yaws)
        if not options:
            raise ApproachFailed("charger approach: no grasp yaw with wrist-limit reserve")
        choice = options[0]
        self._tried_yaws.append(choice.yaw)
        self._event("grasp_target", yaw_deg=math.degrees(choice.yaw),
                    predicted_wrist_margin_deg=math.degrees(choice.margin), q7=self.q7())
        # A vertical grasp is in the allowed tilt range and makes the wrist-roll
        # prediction less sensitive to the orientation inherited from the handle.
        return P._down_quat(choice.yaw, device=self.device)

    def servo(self, target, quat, gripper, **kwargs):
        name = kwargs.get("name", "")
        if name.startswith("charger pre-grasp") or name in ("over charger", "charger descend"):
            if name == "charger descend":
                # The legacy in-place shortcut checks position, not orientation.
                # Finish alignment here before allowing any downward motion.
                yield from self._approach_servo(self.tcp_pose()[0], quat, gripper,
                    pos_tol=0.008, rot_tol=P.EXPERT_GRASP_ROT_TOL,
                    speed=self.speed_careful, max_steps=180, name="charger align")
            return (yield from self._approach_servo(target, quat, gripper, **kwargs))
        return (yield from super().servo(target, quat, gripper, **kwargs))

    def _approach_servo(self, target, quat, gripper, *, speed=None, pos_tol=0.004,
                        rot_tol=0.05, max_steps=250, anchor=None, name="",
                        fail_over=P.STALL_FAIL_M):
        # Strict for grasp approach only: an exhausted/stalled move cannot fall
        # through to closing the fingers, even with a small positional residual.
        speed = self.speed_transit if speed is None else speed
        best_pos = best_rot = float("inf")
        pos_idle = rot_idle = 0
        pn = rn = float("inf")
        for _ in range(max_steps):
            pos, q = self.tcp_pose()
            tp = target() if callable(target) else target
            dp = tp - (anchor() if anchor is not None else pos)
            dr = (torch.zeros(3, device=self.device) if quat is None else
                  self._rot_err(quat() if callable(quat) else quat, q))
            pn, rn = float(torch.linalg.norm(dp)), float(torch.linalg.norm(dr))
            if pn < pos_tol and rn < rot_tol:
                return True
            pos_idle = 0 if pn < best_pos - 0.001 else pos_idle + 1
            rot_idle = 0 if rn < best_rot - 0.005 else rot_idle + 1
            best_pos, best_rot = min(best_pos, pn), min(best_rot, rn)
            q7 = self.q7()
            at_limit = min(q7 - self._q7_limits[0], self._q7_limits[1] - q7) < 0.05
            if rn >= rot_tol and (rot_idle >= 60 or (at_limit and rot_idle >= 25)):
                raise ApproachFailed(f"{name}: rotation stalled ({math.degrees(rn):.1f} deg, q7={q7:.4f})")
            if pn >= pos_tol and pos_idle >= 60:
                raise ApproachFailed(f"{name}: position stalled ({pn * 1000:.1f} mm)")
            yield self._action(dp, dr, gripper, speed)
        raise ApproachFailed(f"{name}: alignment budget exhausted ({pn * 1000:.1f} mm, "
                             f"{math.degrees(rn):.1f} deg)")

    def _pick_charger(self):
        self._tried_yaws = []
        for attempt in range(3):
            try:
                return (yield from super()._pick_charger())
            except ApproachFailed as exc:
                self._event("approach_failed", attempt=attempt + 1, reason=str(exc))
                if attempt == 2:
                    raise P.PlanFailed(f"charger approach: recovery exhausted; {exc}") from exc
                self._event("approach_retry", attempt=attempt + 2)
                # Clearance is vertical and local, not a return to the start pose.
                # Do not move down while choosing another orientation.
                pos, q = self.tcp_pose()
                clear = pos.clone()
                clear[2] = max(float(pos[2]), min(float(pos[2]) + 0.06,
                                                float(self.charger_center()[2]) + 0.065))
                yield from self._approach_servo(clear, q, P.OPEN, pos_tol=0.008,
                    rot_tol=0.15, speed=self.speed_careful, max_steps=90, name="charger recovery clearance")
        return False
