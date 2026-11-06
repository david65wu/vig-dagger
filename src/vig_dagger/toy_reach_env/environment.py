"""Single-scene imitation-learning environment with explicit pose initialization."""

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import compute_pose_error

from . import mdp


class ToyReachEnv(ManagerBasedEnv):
    def __init__(self, cfg):
        if cfg.scene.num_envs != 1:
            raise ValueError("ToyReach currently supports one environment for controlled camera probes.")
        if cfg.tube_radius < 0 or cfg.goal_tolerance <= 0:
            raise ValueError("tube_radius must be nonnegative and goal_tolerance positive")
        if not all(torch.isfinite(torch.tensor(p)).all() for p in (cfg.point_a, cfg.point_b)):
            raise ValueError("A and B must be finite")
        if torch.linalg.vector_norm(torch.tensor(cfg.point_b) - torch.tensor(cfg.point_a)) < 1e-6:
            raise ValueError("A and B must be distinct")
        for name, point in (("start", cfg.point_a), ("goal", cfg.point_b)):
            getattr(cfg.scene, f"{name}_marker").init_state.pos = tuple(point)
            getattr(cfg.scene, f"{name}_projection").init_state.pos = (point[0], point[1], 0.012)
        super().__init__(cfg)

    def reset(self, seed=None, env_ids=None, options=None):
        super().reset(seed=seed, env_ids=env_ids, options=options)
        self.scene["third_person_cam"].set_world_poses_from_view(
            torch.tensor([self.cfg.camera_eye], device=self.device),
            torch.tensor([self.cfg.camera_target], device=self.device),
        )
        options = options or {}
        start = options.get("eef_position")
        if start is None:
            a = torch.tensor(self.cfg.point_a, device=self.device)
            direction = torch.tensor(self.cfg.point_b, device=self.device) - a
            direction /= direction.norm()
            # Isotropic Gaussian direction projected into the normal plane; uniform disk area.
            offset = torch.randn(3, device=self.device)
            offset -= offset.dot(direction) * direction
            offset /= offset.norm().clamp_min(1e-12)
            radius = self.cfg.tube_radius if options.get("randomize_start", True) else 0.0
            start = a + offset * torch.rand((), device=self.device).sqrt() * radius
        obs = self.place_eef(start)
        self.action_manager.reset()
        self.observation_manager.reset()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.extras["start_position"] = obs["policy"][0, :3].clone()
        return self.obs_buf, self.extras

    def action_toward(self, position, max_translation=0.01):
        """Pose servo for setup and smoke checks, not a demonstration planner.

        Translation commands are meters and rotation commands are radians, with no
        OpenDrawer scale factor. Positive gripper command keeps the fingers open.
        """
        pose = mdp.tcp_pose(self)
        target = torch.as_tensor(position, device=self.device, dtype=pose.dtype).reshape(1, 3)
        orientation = torch.tensor([self.cfg.eef_quaternion], device=self.device, dtype=pose.dtype)
        dp, dr = compute_pose_error(pose[:, :3], pose[:, 3:], target, orientation, rot_error_type="axis_angle")
        dp *= (max_translation / dp.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1)
        dr *= (0.05 / dr.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1)
        return torch.cat((dp, dr, torch.ones((1, 1), device=self.device)), dim=-1)

    @torch.inference_mode()
    def place_eef(self, position, max_steps=1800, tolerance=0.001):
        """Physically settle at a reachable probe pose, then refresh both cameras.

        Does not emit episode steps or recorder samples. Raises if IK cannot reach
        the requested pose. Intended for reset and independent OOD observations.
        """
        target = torch.as_tensor(position, dtype=torch.float32, device=self.device).reshape(1, 3)
        if not torch.isfinite(target).all():
            raise ValueError("EEF position must be finite")
        stable = 0
        for _ in range(max_steps):
            self.action_manager.process_action(self.action_toward(target))
            for _ in range(self.cfg.decimation):
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
            pose = mdp.tcp_pose(self)
            pos_error = (pose[:, :3] - target).norm()
            quat = torch.tensor([self.cfg.eef_quaternion], device=self.device)
            _, rot_error = compute_pose_error(pose[:, :3], pose[:, 3:], target, quat, rot_error_type="axis_angle")
            settled = (pos_error < tolerance and rot_error.norm() < 0.01
                       and self.scene["robot"].data.joint_vel.abs().max() < 0.02)
            stable = stable + 1 if settled else 0
            if stable >= 5:
                break
        else:
            raise RuntimeError(
                f"Could not settle EEF at {target.tolist()}; position error {float(pos_error):.5f} m, "
                f"rotation error {rot_error.tolist()}, actual pose {pose.tolist()}, "
                f"joints {self.scene['robot'].data.joint_pos.tolist()}, "
                f"velocities {self.scene['robot'].data.joint_vel.tolist()}"
            )
        for _ in range(3):
            self.sim.render()
        for name in ("third_person_cam", "wrist_cam"):
            self.scene[name].update(self.physics_dt, force_recompute=True)
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf
