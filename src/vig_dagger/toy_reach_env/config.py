"""A minimal two-camera scene for controlled spatial OOD experiments."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnvCfg, mdp as lab_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp


def marker(path, position, color):
    return AssetBaseCfg(
        prim_path=path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.SphereCfg(radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
    )


@configclass
class ToyReachSceneCfg(InteractiveSceneCfg):
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(
        prim_path="/World/Ground", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.06)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.0, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.36, 0.40), roughness=0.8),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    start_marker = marker("{ENV_REGEX_NS}/Start", (0.45, -0.15, 0.35), (0.12, 0.35, 0.95))
    goal_marker = marker("{ENV_REGEX_NS}/Goal", (0.45, 0.15, 0.35), (0.1, 0.85, 0.25))
    start_projection = marker("{ENV_REGEX_NS}/StartProjection", (0.45, -0.15, 0.012), (0.12, 0.35, 0.95))
    goal_projection = marker("{ENV_REGEX_NS}/GoalProjection", (0.45, 0.15, 0.012), (0.1, 0.85, 0.25))
    third_person_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPersonCamera", height=256, width=256, update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.15, 0.9, 0.85), convention="world"),
    )
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera", height=256, width=256, update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.01, 3.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.08, 0, -0.05), rot=(0.08002, 0.70256, 0.70256, 0.08002), convention="opengl"),
    )


@configclass
class ActionsCfg:
    # Seven commands: base-frame delta xyz, rotation vector xyz, binary gripper.
    arm = lab_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["panda_joint[1-7]"], body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0, body_offset=lab_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=mdp.TCP_OFFSET),
    )
    gripper = lab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04}, close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        ee_pose = ObservationTermCfg(func=mdp.tcp_pose)
        gripper_qpos = ObservationTermCfg(func=mdp.fingers)
        concatenate_terms = True
        enable_corruption = False

    @configclass
    class ImagesCfg(ObservationGroupCfg):
        third_person_cam = ObservationTermCfg(
            func=lab_mdp.image, params={"sensor_cfg": SceneEntityCfg("third_person_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObservationTermCfg(
            func=lab_mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        concatenate_terms = False
        enable_corruption = False

    @configclass
    class TaskCfg(ObservationGroupCfg):
        goal_distance = ObservationTermCfg(func=mdp.goal_distance)
        line_distance = ObservationTermCfg(func=mdp.line_distance)
        success = ObservationTermCfg(func=mdp.success)
        concatenate_terms = False
        enable_corruption = False

    policy = PolicyCfg()
    rgb_camera = ImagesCfg()
    task = TaskCfg()


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=lab_mdp.reset_scene_to_default, mode="reset")


@configclass
class ToyReachEnvCfg(ManagerBasedEnvCfg):
    scene = ToyReachSceneCfg(num_envs=1, env_spacing=2.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    point_a: tuple[float, float, float] = (0.45, -0.15, 0.35)
    point_b: tuple[float, float, float] = (0.45, 0.15, 0.35)
    eef_quaternion: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    tube_radius: float = 0.01
    goal_tolerance: float = 0.005
    camera_eye: tuple[float, float, float] = (1.15, 0.9, 0.85)
    camera_target: tuple[float, float, float] = (0.35, 0, 0.25)

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.device = "cpu"
        self.num_rerenders_on_reset = 2
        self.viewer.eye = self.camera_eye
        self.viewer.lookat = self.camera_target
