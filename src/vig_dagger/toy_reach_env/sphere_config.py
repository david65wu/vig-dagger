"""Random sphere-endpoint reaching with an explicit goal observation."""
import torch
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass

from .config import ObservationsCfg, ToyReachEnvCfg


def goal_position(env):
    return torch.tensor(env.cfg.point_b, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)


@configclass
class SphereObservationsCfg(ObservationsCfg):
    @configclass
    class GoalPolicyCfg(ObservationsCfg.PolicyCfg):
        goal_xyz = ObservationTermCfg(func=goal_position)

    policy = GoalPolicyCfg()


@configclass
class SphereReachEnvCfg(ToyReachEnvCfg):
    observations = SphereObservationsCfg()
    sphere_center: tuple[float, float, float] = (.45, 0., .35)
    sphere_radius: float = .15
    min_endpoint_separation: float = .10
    tube_radius: float = .008
    endpoint_mode: str = 'random_pair'
    fixed_goal_direction: tuple[float, float, float] = (0., 0., 1.)
