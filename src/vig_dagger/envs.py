"""The PlugCharger task: `PlugCharger-Hole4mm-SideCam-RandSocket-v1`.

ManiSkill's `PlugCharger-v1` (pick up the charger, insert its two pins into the socket) with four changes, all
made in one class here (the original repo spread them over a chain of subclasses with the same overrides):

- **4 mm clearance per side** in the socket holes (stock: 0.5 mm), so a planner can insert reliably.
- **Two more fixed cameras.** `third_person_camera` (side-on view of the whole workspace, fixed to the table) and
  `side_camera` (26 cm from the socket face, about 70 degrees off the insertion axis, at socket height), next to the
  stock `base_camera` and the wrist-mounted `hand_camera`. Policies use hand, third-person and side cameras.
- **The socket moves per episode:** up to +-3 cm sideways, +-2 cm in height and +-5 degrees in yaw around its stock
  pose, so no fixed joint path solves the task and the policy has to align from vision. The goal moves with it.
- **Charcoal-gray socket body** (stock: white like the charger), so socket and plug are easy to tell apart.

Success (inherited): the plug within 5 mm and 0.2 rad of the goal pose. The registered step limit is 300 (the
planner's demos fit in it); DAgger raises it to 400 (`max_episode_steps`).

The socket draws come after the stock charger draws, so a scene seed places the charger exactly as in stock
PlugCharger. Importing this module registers the environment.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

ENV_ID = "PlugCharger-Hole4mm-SideCam-RandSocket-v1"
POLICY_CAMERAS = ("hand_camera", "third_person_camera", "side_camera")

SOCKET_FACE = [0.045, 0.0, 0.10]  # receptacle face centre at the stock socket pose
SIDE_CAMERA_EYE = [-0.044, 0.244, 0.12]  # 26 cm from the face, ~70 deg off the insertion axis, socket height
SIDE_CAMERA_FOV = 0.55


@register_env(ENV_ID, max_episode_steps=300)
class PlugChargerRandSocketEnv(PlugChargerEnv):
    _clearance = 4e-3  # single-side hole clearance (stock: 5e-4)

    SOCKET_HALF_RANGE = dict(y=0.03, z=0.02, yaw_deg=5.0)
    SOCKET_COLOR = "#55585C"

    @property
    def _default_sensor_configs(self):
        configs = list(super()._default_sensor_configs)
        # Fixed to the table (no mount), so it doesn't move with the socket. Clock-face framing seen from above
        # (robot base at 3 o'clock, socket at 9): placed at 5 o'clock, facing 11, about at charger height.
        configs.append(CameraConfig(
            "third_person_camera", pose=sapien_utils.look_at(eye=[-0.26, 0.3637, 0.06], target=[-0.05, 0, 0.03]),
            width=512, height=512, fov=1))
        # The socket view: once the plug touches the face, the wrist camera sees a blank plane and the third-person
        # camera is too far to resolve a 4 mm miss. Both holes stay visible through the approach from here.
        configs.append(CameraConfig(
            "side_camera", pose=sapien_utils.look_at(eye=SIDE_CAMERA_EYE, target=SOCKET_FACE),
            width=512, height=512, fov=SIDE_CAMERA_FOV))
        return configs

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        color = sapien_utils.hex2rgba(self.SOCKET_COLOR)
        for obj in self.receptacle._objs:
            for shape in obj.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
                if np.allclose(shape.material.base_color, [1, 1, 1, 1]):  # the white body; gold holes stay
                    shape.material.base_color = color

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            r = self.SOCKET_HALF_RANGE
            yaw = np.deg2rad(r["yaw_deg"])
            d = randomization.uniform([-r["y"], -r["z"], -yaw], [r["y"], r["z"], yaw], size=(b, 3))
            pos = torch.zeros((b, 3))
            pos[:, 0] = 0.055
            pos[:, 1] = d[:, 0]
            pos[:, 2] = 0.1 + d[:, 1]
            ori = torch.stack([common.to_tensor(euler2quat(0, 0, np.pi + float(a))) for a in d[:, 2].cpu()])
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))
            self.goal_pose = self.receptacle.pose * sapien.Pose(q=euler2quat(0, 0, np.pi))


def make_env(*, max_episode_steps: int | None = None, cam_size: int = 256, obs_mode: str = "rgb",
             sim_backend: str = "cpu"):
    """The task with joint-position control and square camera images, as the demos and DAgger use it."""
    import gymnasium as gym

    kwargs = {} if max_episode_steps is None else {"max_episode_steps": int(max_episode_steps)}
    return gym.make(
        ENV_ID, num_envs=1, obs_mode=obs_mode, control_mode="pd_joint_pos", render_mode="rgb_array",
        reward_mode="sparse", sensor_configs=dict(shader_pack="default", width=cam_size, height=cam_size),
        human_render_camera_configs=dict(shader_pack="default"), sim_backend=sim_backend, **kwargs)
