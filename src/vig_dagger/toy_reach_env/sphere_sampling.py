"""Simulator-independent endpoint sampling on a sphere."""
import numpy as np


def sample_fixed_goal_pair(center=(.45, 0., .35), radius=.15, goal_direction=(0., 0., 1.), seed=None):
    """Uniform surface A on the hemisphere opposite a fixed sphere-surface B."""
    center = np.asarray(center, dtype=np.float64)
    direction = np.asarray(goal_direction, dtype=np.float64)
    if center.shape != (3,) or direction.shape != (3,) or not np.isfinite([center, direction]).all():
        raise ValueError('Center and goal direction must be finite 3D vectors')
    norm = np.linalg.norm(direction)
    if norm < 1e-12 or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Goal direction must be nonzero and radius positive')
    direction = direction / norm
    rng = np.random.default_rng(seed)
    sample = rng.normal(size=3)
    while np.linalg.norm(sample) < 1e-12:
        sample = rng.normal(size=3)
    sample /= np.linalg.norm(sample)
    # Reflect the B-facing hemisphere across the equatorial plane. Both source
    # hemispheres have equal area density, so this preserves uniform surface sampling.
    sample -= 2 * max(0., float(sample @ direction)) * direction
    return center + radius * sample, center + radius * direction


def sample_sphere_pair(center=(.45, 0., .35), radius=.15, min_separation=.10, seed=None):
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Sphere center must contain three finite coordinates')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Sphere radius must be positive and finite')
    if not np.isfinite(min_separation) or not 0 < min_separation < 2*radius:
        raise ValueError('Minimum separation must be positive and below the sphere diameter')
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        directions = rng.normal(size=(2, 3))
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        if (lengths < 1e-12).any():
            continue
        points = center + radius*directions/lengths
        if np.linalg.norm(points[1]-points[0]) >= min_separation:
            return points[0], points[1]
    raise RuntimeError('Could not sample sufficiently separated sphere endpoints')
