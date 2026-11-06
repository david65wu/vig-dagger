"""Independent sphere task; the original fixed-endpoint environment is unchanged."""
import numpy as np
import torch

from .environment import ToyReachEnv
from .planner import TubePlan
from .sphere_sampling import sample_sphere_pair, sample_fixed_goal_pair


class SphereReachEnv(ToyReachEnv):
    @staticmethod
    def sample_endpoints(cfg, seed):
        if cfg.endpoint_mode == 'fixed_goal':
            return sample_fixed_goal_pair(cfg.sphere_center, cfg.sphere_radius, cfg.fixed_goal_direction, seed)
        if cfg.endpoint_mode != 'random_pair':
            raise ValueError(f'Unknown endpoint mode: {cfg.endpoint_mode}')
        return sample_sphere_pair(cfg.sphere_center, cfg.sphere_radius, cfg.min_endpoint_separation, seed)

    def __init__(self, cfg):
        a, b = self.sample_endpoints(cfg, seed=0)
        cfg.point_a, cfg.point_b = tuple(a), tuple(b)
        self._endpoint_rng = np.random.default_rng(0)
        super().__init__(cfg)

    def reset(self, seed=None, env_ids=None, options=None):
        options = dict(options or {})
        if ('point_a' in options) != ('point_b' in options):
            raise ValueError('Explicit resets need both point_a and point_b')
        if 'point_a' in options:
            a, b = np.asarray(options.pop('point_a'), dtype=float), np.asarray(options.pop('point_b'), dtype=float)
            for point in (a, b):
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError('Endpoints must contain three finite coordinates')
                if abs(np.linalg.norm(point-np.asarray(self.cfg.sphere_center))-self.cfg.sphere_radius) > 1e-6:
                    raise ValueError('Both endpoints must lie on the configured sphere')
            if np.linalg.norm(b-a) < self.cfg.min_endpoint_separation:
                raise ValueError('Endpoints are too close')
        else:
            endpoint_seed = seed if seed is not None else int(self._endpoint_rng.integers(0, 2**31))
            a, b = self.sample_endpoints(self.cfg, endpoint_seed)
        self.cfg.point_a, self.cfg.point_b = tuple(a), tuple(b)
        for name, point in [('start', a), ('goal', b)]:
            self.scene[f'{name}_marker'].set_world_poses(
                positions=torch.tensor(point, device=self.device, dtype=torch.float32)[None]+self.scene.env_origins)
            projection = torch.tensor([point[0], point[1], .012], device=self.device, dtype=torch.float32)[None]
            self.scene[f'{name}_projection'].set_world_poses(positions=projection+self.scene.env_origins)
        # A is exactly on the sphere; path variations affect interior control points.
        options.setdefault('eef_position', a)
        obs, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        extras['point_a'] = torch.tensor(a, device=self.device, dtype=torch.float32)[None]
        extras['point_b'] = torch.tensor(b, device=self.device, dtype=torch.float32)[None]
        return obs, extras

    def sample_plan(self, seed):
        plan = TubePlan.sample(self.cfg.point_a, self.cfg.point_b, self.cfg.tube_radius, seed)
        plan.control_points[0] = plan.point_a
        return plan
