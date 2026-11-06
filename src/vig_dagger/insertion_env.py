"""Insertion-only resets around the unchanged main-branch PlugCharger task."""
from dataclasses import replace

import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from vig_dagger.envs import ENV_ID
from vig_dagger.expert import planner as expert


def make_planner(env):
    return PandaArmMotionPlanningSolver(env, debug=False, vis=False,
        base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=False,
        print_env_info=False, joint_vel_limits=1.0, joint_acc_limits=1.0)


def checked(result, stage):
    if expert._failed(result):raise RuntimeError(f'Motion planning failed during {stage}')
    return result


class InsertionResetWrapper(gym.Wrapper):
    """Initialize a real contact grasp off-record, then reset the episode clock.

    Charger/socket physics, action space, cameras and success condition remain those
    of the wrapped environment. No weld, kinematic attachment, or pose teleport is
    used to hold the charger during the recorded episode.
    """
    STANDOFF = (.03, .08)  # charger root relative to its seated goal
    LATERAL = .015
    ANGLE_DEG = 8.

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        evaluation_pose = options.pop('insertion_pose', None)
        _, info = self.env.reset(seed=seed, options=options)
        base = self.unwrapped
        start_seed = 900000 + int(seed or 0)
        rng = np.random.default_rng(start_seed)
        distance = float(rng.uniform(*self.STANDOFF))
        lateral = rng.uniform(-self.LATERAL, self.LATERAL, size=2)
        angles = rng.uniform(-self.ANGLE_DEG, self.ANGLE_DEG, size=3)
        if evaluation_pose is not None:
            distance = float(evaluation_pose['standoff_m'])
            lateral = np.asarray(evaluation_pose['lateral_yz_m'], dtype=float)
            angles = np.asarray(evaluation_pose['rpy_deg'], dtype=float)
            if lateral.shape != (2,) or angles.shape != (3,) or not np.isfinite(np.r_[distance,lateral,angles]).all():
                raise ValueError('Invalid insertion evaluation pose')
        params = replace(expert.StrategyParams.sample(start_seed), grasp_tilt_deg=15.,
            grasp_offset=0., grasp_flip=False, lift=True, lift_height=.06, lift_dx=0., lift_dy=0.,
            speed_transit=.6, speed_insert=.5, transit_planner='screw')
        planner = make_planner(self.env)
        try:
            expert._init_speed(planner); expert._set_speed(planner, params.speed_transit)
            grasp = expert._grasp_pose(base, params)
            checked(expert._transit(planner, grasp * sapien.Pose([0, 0, -expert.REACH_HEIGHT]), params), 'setup reach')
            checked(planner.move_to_pose_with_screw(grasp), 'setup grasp approach')
            planner.close_gripper()
            checked(expert._transit(planner, sapien.Pose([0, 0, .06]) * grasp, params), 'setup lift')
            if not bool(base.agent.is_grasping(base.charger)[0]):
                raise RuntimeError('Setup failed to establish a physical charger grasp')
            tcp_in_charger = base.charger.pose.sp.inv() * base.agent.tcp.pose.sp
            requested_relative = sapien.Pose([-distance, *lateral], euler2quat(*np.deg2rad(angles)))
            requested_charger = base.goal_pose.sp * requested_relative
            checked(expert._transit(planner, requested_charger * tcp_in_charger, params), 'near-socket initialization')
            checked(planner.move_to_pose_with_screw(requested_charger * tcp_in_charger, refine_steps=10), 'initialization settle')
            held = []
            hold_action = np.hstack([base.agent.robot.get_qpos()[0, :7].cpu().numpy(), planner.CLOSED])
            for _ in range(10):
                self.env.step(hold_action)
                held.append(bool(base.agent.is_grasping(base.charger)[0]))
            if not all(held[-5:]):raise RuntimeError('Initial near-socket grasp is not stable')
            relative = base.goal_pose.sp.inv() * base.charger.pose.sp
            actual_angles = np.rad2deg(quat2euler(relative.q))
            position_error = float(np.linalg.norm(relative.p - requested_relative.p))
            rotation_error = float(2*np.arccos(np.clip(abs(np.dot(relative.q, requested_relative.q)), 0, 1)))
            if position_error > .003 or rotation_error > np.deg2rad(2):
                raise RuntimeError(f'Initialization tracking error: {position_error} m, {rotation_error} rad')
            if bool(base.evaluate()['success'][0]):raise RuntimeError('Insertion reset must not already be successful')
            self.initialization = dict(scene_seed=seed, initialization_seed=start_seed,
                requested_standoff_m=distance, requested_lateral_yz_m=lateral.tolist(),
                requested_rpy_deg=angles.tolist(), actual_goal_relative_xyz_m=relative.p.tolist(),
                actual_goal_relative_rpy_deg=actual_angles.tolist(), position_error_m=position_error,
                rotation_error_rad=rotation_error, held=True, setup_steps=int(base.elapsed_steps[0]),
                charger_pose=base.charger.pose.raw_pose[0].cpu().tolist(),
                tcp_pose=base.agent.tcp.pose.raw_pose[0].cpu().tolist(),
                goal_pose=base.goal_pose.raw_pose[0].cpu().tolist(),
                socket_pose=base.receptacle.pose.raw_pose[0].cpu().tolist(),
                grasp='Main-branch nominal 15-degree centered grasp; physical contacts, no attachment')
        finally:
            planner.close()
        # Reset only clocks. The robot, charger, socket, contact state, and control
        # targets stay at the physically established near-socket starting state.
        base._elapsed_steps.zero_()
        cursor = self.env
        while isinstance(cursor, gym.Wrapper):
            if '_elapsed_steps' in cursor.__dict__:cursor._elapsed_steps = 0
            cursor = cursor.env
        info = dict(info, **base.get_info())
        return base.get_obs(info), dict(info, insertion_initialization=self.initialization)


def make_insertion_env(*, render_backend, cam_size=256, max_episode_steps=300):
    env = gym.make(ENV_ID, num_envs=1, obs_mode='rgb', control_mode='pd_joint_pos',
        render_mode='rgb_array', reward_mode='sparse', sim_backend='cpu',
        render_backend=render_backend, max_episode_steps=max_episode_steps,
        sensor_configs=dict(shader_pack='default', width=cam_size, height=cam_size),
        human_render_camera_configs=dict(shader_pack='default'))
    return InsertionResetWrapper(env)


def solve_insertion(env, state_log):
    """Main expert's align / refine / straight insert / closed-hold stages only."""
    base = env.unwrapped
    if not bool(base.agent.is_grasping(base.charger)[0]):raise RuntimeError('Insertion starts without a grasp')
    planner = make_planner(env); boundaries=[]
    try:
        expert._init_speed(planner); expert._set_speed(planner, .5)
        planner.gripper_state = planner.CLOSED
        tcp_in_charger = base.charger.pose.sp.inv() * base.agent.tcp.pose.sp
        relative = base.goal_pose.sp.inv() * base.charger.pose.sp
        distance = max(float(-relative.p[0]), expert.SOCKET_FACE_DEPTH + expert.ALIGN_CLEARANCE)
        pre_insert = base.goal_pose.sp * sapien.Pose([-distance, 0, 0]) * tcp_in_charger
        insert = base.goal_pose.sp * tcp_in_charger
        def stage(name, call):
            start=state_log.num_steps
            result=checked(call(), name)
            boundaries.append(dict(stage=name,start_step=start,end_step=state_log.num_steps))
            return result
        stage('align', lambda: planner.move_to_pose_with_screw(pre_insert))
        stage('refine', lambda: planner.move_to_pose_with_screw(pre_insert, refine_steps=5))
        stage('insert', lambda: planner.move_to_pose_with_screw(insert))
        result=stage('hold', lambda: planner.close_gripper(t=expert.PAUSE_STEPS))
        return result, boundaries
    finally:
        planner.close()
