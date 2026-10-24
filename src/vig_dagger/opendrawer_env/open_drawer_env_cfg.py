# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim import schemas as sim_schemas
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.utils import bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from pxr import Usd, UsdGeom, UsdPhysics  # isort: skip


CHARGER_USD = Path(__file__).resolve().parent / "assets" / "charger001" / "model_charger.usd"

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@clone
def spawn_charger_as_rigid(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a visual USD and author the physics schemas needed by RigidObjectCfg."""
    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(
            prim_path,
            usd_path=cfg.usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )

    root_prim = stage.GetPrimAtPath(prim_path)
    rigid_props = cfg.rigid_props or RigidBodyPropertiesCfg()
    collision_props = cfg.collision_props or CollisionPropertiesCfg(collision_enabled=True)

    rigid_body_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(root_prim) if UsdPhysics.RigidBodyAPI(prim)]
    if rigid_body_paths:
        rigid_body_path = rigid_body_paths[0]
        sim_schemas.modify_rigid_body_properties(rigid_body_path, rigid_props, stage=stage)
    else:
        rigid_body_path = prim_path
        sim_schemas.define_rigid_body_properties(rigid_body_path, rigid_props, stage=stage)

    if cfg.mass_props is not None:
        sim_schemas.define_mass_properties(rigid_body_path, cfg.mass_props, stage=stage)

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            sim_schemas.define_collision_properties(str(prim.GetPath()), collision_props, stage=stage)

    return root_prim


@clone
def spawn_table_with_color(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a table USD and apply a color override, bypassing instanceable geometry."""
    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(
            prim_path,
            usd_path=cfg.usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )

    root_prim = stage.GetPrimAtPath(prim_path)

    # Deinstance all child prims so material bindings can reach the geometry.
    for prim in Usd.PrimRange(root_prim):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)

    if cfg.visual_material is not None:
        material_path = f"{prim_path}/material"
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(prim_path, material_path, stage=stage)

    return root_prim


@configclass
class OpenDrawerSceneCfg(InteractiveSceneCfg):
    """Scene with a Franka Panda, charger, side table, and Sektion cabinet drawer."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.75,
                "panda_joint3": 0.0,
                "panda_joint4": -2.45,
                "panda_joint5": 0.0,
                "panda_joint6": 2.35,
                "panda_joint7": 2.312,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )

    left_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.42, 0.42), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_table_with_color,
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.06, 0.02), roughness=0.8),
        ),
    )

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
            scale=(0.65, 0.65, 0.65),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.06, 0.02), roughness=0.72),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.9, 0.0, 0.26),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=0.0,
                friction=5.0,
                dynamic_friction=2.0,
                viscous_friction=0.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    charger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Charger",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.42, 0.445), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_charger_as_rigid,
            usd_path=str(CHARGER_USD),
            scale=(1.0, 1.0, 1.0),
            mass_props=MassPropertiesCfg(mass=0.08),
            collision_props=CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="gripper",
            ),
        ],
    )

    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),
                ),
            ),
        ],
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    teleop_cam: CameraCfg = MISSING
    left_cam: CameraCfg = MISSING
    right_cam: CameraCfg = MISSING
    wrist_cam: CameraCfg = MISSING
    top_cam: CameraCfg = MISSING


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation groups for control, dataset pose labels, and camera streams."""

    @configclass
    class LowDimPolicyCfg(ObsGroup):
        """Low-dimensional observations consumed by the policy/action loop."""

        # Robot proprioception.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Drawer state.
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )

        # Task object state and task-relative geometry.
        charger_pos = ObsTerm(func=mdp.charger_pos)
        rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)
        rel_ee_charger_distance = ObsTerm(func=mdp.rel_ee_charger_distance)
        rel_charger_drawer_distance = ObsTerm(func=mdp.rel_charger_drawer_distance)

        # Previous action for temporal context. The recorder still saves commanded actions under dataset key "actions".
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Keep the group name as "policy" for Isaac Lab compatibility and existing recorder keys.
    policy: LowDimPolicyCfg = LowDimPolicyCfg()

    @configclass
    class PoseCfg(ObsGroup):
        """Explicit 7D poses saved to obs/pose for dataset consumers."""

        ee_pose = ObsTerm(func=mdp.ee_pose)
        gripper_pose = ObsTerm(func=mdp.gripper_pose)
        cabinet_pose = ObsTerm(func=mdp.cabinet_pose)
        left_table_pose = ObsTerm(func=mdp.left_table_pose)
        charger_pose = ObsTerm(func=mdp.charger_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    pose: PoseCfg = PoseCfg()

    @configclass
    class RGBCameraCfg(ObsGroup):
        """RGB image streams saved to obs/rgb_camera for dataset consumers."""

        teleop_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("teleop_cam"), "data_type": "rgb", "normalize": False},
        )
        left_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_cam"), "data_type": "rgb", "normalize": False},
        )
        right_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        top_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    rgb_camera: RGBCameraCfg = RGBCameraCfg()


@configclass
class EventCfg:
    """Configuration for scene randomization and reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cabinet_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.13), "y": (-0.12, 0.12), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cabinet"),
        },
    )

    reset_charger_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.12, 0.08), "y": (-0.12, 0.12), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("charger"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": 0.04})
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": 0.04,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_.*"]),
        },
    )
    open_drawer_bonus = RewTerm(
        func=mdp.open_drawer_bonus,
        weight=7.5,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )
    multi_stage_open_drawer = RewTerm(
        func=mdp.multi_stage_open_drawer,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )
    approach_ee_charger = RewTerm(func=mdp.approach_ee_charger, weight=2.0, params={"threshold": 0.15})
    lift_charger = RewTerm(func=mdp.charger_lifted, weight=4.0, params={"minimal_height": 0.12})
    charger_to_drawer = RewTerm(
        func=mdp.charger_to_drawer,
        weight=6.0,
        params={"drawer_open_threshold": 0.18, "target_offset": (-0.16, 0.0, -0.04)},
    )
    charger_in_drawer = RewTerm(
        func=mdp.charger_in_drawer_bonus,
        weight=8.0,
        params={
            "threshold": 0.18,
            "target_offset": (-0.16, 0.0, -0.04),
            "zone_half_extents": (0.34, 0.28, 0.22),
        },
    )
    close_drawer_after_charger = RewTerm(
        func=mdp.close_drawer_after_charger,
        weight=10.0,
        params={
            "charger_threshold": 0.18,
            "drawer_target_offset": (-0.16, 0.0, -0.04),
            "zone_half_extents": (0.34, 0.28, 0.22),
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    success = DoneTerm(
        func=mdp.drawer_closed_with_charger,
        params={
            "charger_threshold": 0.18,
            "drawer_closed_threshold": 0.01,
            "drawer_target_offset": (-0.30, 0.0, 0.0),
            "zone_half_extents": (0.34, 0.28, 0.22),
            "placement_zone_half_extents": (0.16, 0.21, 0.03),
            "drawer_open_threshold": 0.18,
            "pickup_height_delta": 0.04,
            "print_success_stages": True,
            "print_stage3_box_debug": False,
            "stage3_debug_interval": 15,
            "draw_stage3_box_debug": False,
            "draw_randomization_debug": False,
            "charger_randomization_center": (0.28, 0.42, 0.445),
            "charger_randomization_half_extents": (0.10, 0.12, 0.01),
            "cabinet_randomization_center": (0.94, 0.0, 0.78),
            "cabinet_randomization_half_extents": (0.09, 0.12, 0.01),
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class OpenDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based Franka open-drawer environment with relative EEF IK actions."""

    scene: OpenDrawerSceneCfg = OpenDrawerSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        self.scene.teleop_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/teleop_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, clipping_range=(0.01, 4.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.01238, 0.97199, 1.18626),
                rot=(0.23118657, 0.10514346, -0.40041997, -0.88043255),
                convention="opengl",
            ),
        )
        self.scene.left_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/left_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=32.0, clipping_range=(0.01, 4.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.42308, 1.49076, 1.24692),
                rot=(0.00445859, 0.00266447, 0.51297584, 0.85838733),
                convention="opengl",
            ),
        )
        self.scene.right_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/right_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=30.0, clipping_range=(0.01, 4.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.56424, -0.56924, 1.61692),
                rot=(0.97099145, 0.23911410, 0.00005438, 0.00022082),
                convention="opengl",
            ),
        )
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, clipping_range=(0.01, 2.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.08, 0.0, -0.05),
                # rot=(-0.70106, 0.07985, 0.07985, -0.70106),
                rot=(0.08002, 0.70256, 0.70256, 0.08002),
                # pos=(0.0, 0.0, 0.0),
                # rot=(1.0, 0.0, 0.0, 0.0),
                convention="opengl",
            ),
        )
        self.scene.top_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/top_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=27.0, clipping_range=(0.01, 4.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.45484, 0.28785, 1.54408),
                rot=(0.707339, -0.026958, -0.026901, -0.705848),
                convention="opengl",
            )
        )
        self.image_obs_list = ["teleop_cam", "left_cam", "right_cam", "wrist_cam", "top_cam"]

