"""An interruptible ``PandaArmMotionPlanningSolver`` for handing control back to the policy mid-plan.

Stock ``follow_path`` steps the env through every waypoint of a planned path with no hook for the caller. During a
DAgger takeover the query score keeps being computed after every expert step, and once the gate decides to return
control (see ``vig_dagger/gate.py``) the rest of the plan must be dropped.

``InterruptiblePandaArmMotionPlanningSolver`` adds an ``on_step`` callback, invoked after every ``env.step()``
inside ``follow_path`` (the only place that steps the env during ``move_to_pose_with_screw`` / ``_RRT*``). Once the
callback returns True, ``self.aborted`` latches and all remaining waypoints of that and later ``follow_path`` calls
are skipped; the expert's stage sequence stops issuing moves via ``AbortedError``.

``open_gripper``/``close_gripper`` are not interruptible (letting go mid-close would drop the charger) and don't
call ``on_step``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

# (obs, reward, terminated, truncated, info, action) -> True to abort the remaining path.
StepCallback = Callable[[dict, object, object, object, dict, np.ndarray], bool]


class AbortedError(Exception):
    """Raised by resumable solvers to unwind out of a stage sequence once ``planner.aborted`` is set."""


class InterruptiblePandaArmMotionPlanningSolver(PandaArmMotionPlanningSolver):
    def __init__(self, *args, on_step: StepCallback | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_step = on_step
        self.aborted = False
        self._last_step_result = None
        # While True, an abort signalled by on_step is *latched* (self.aborted)
        # but not acted on: the current and subsequent paths keep executing
        # (without polling on_step again -- the DAgger state machine has
        # already flipped back to policy mode, so it must not see further
        # expert steps) until the solver clears the flag and calls
        # check_aborted(). Solvers use it to make an atomic unit (e.g.
        # reach -> descend -> close -> lift) finish before control is handed
        # back, instead of returning a half-completed grasp to the policy.
        self.defer_abort = False

    def follow_path(self, result, refine_steps: int = 0):
        if self.aborted and not self.defer_abort:
            return self._last_step_result

        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            step_result = self.env.step(action)
            self.elapsed_steps += 1
            self._last_step_result = step_result
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={step_result[1]} info={step_result[4]}")
            if self.vis:
                self.base_env.render_human()
            if self._on_step is not None and not self.aborted and self._on_step(*step_result, action):
                self.aborted = True
                if not self.defer_abort:
                    break
        return self._last_step_result

    def check_aborted(self) -> None:
        """Call after every planner motion in a resumable solver's stage sequence.
        A latched abort is only raised once ``defer_abort`` is cleared."""
        if self.aborted and not self.defer_abort:
            raise AbortedError()
