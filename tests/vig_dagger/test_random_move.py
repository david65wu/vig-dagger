import numpy as np

from vig_dagger.expert import random_move as rm


def test_offsets_stay_in_the_ball():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d, axis, angle = rm.sample_offset(rng, 0.05, 5.0)
        assert np.linalg.norm(d) <= 0.05 + 1e-12
        assert abs(np.linalg.norm(axis) - 1) < 1e-9 and 0 <= angle <= np.deg2rad(5.0)


def _geometry():
    """A 2 cm cube of moving points at the origin; an obstacle box (a wall) 3 cm to its +x side; table far below."""
    g = np.linspace(-0.01, 0.01, 5)
    pts = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3) + [0, 0, 0.5]
    wall = (np.eye(4), np.array([0.005, 0.1, 0.1]))
    wall[0][:3, 3] = [0.045, 0, 0.5]
    return dict(mov_pts=pts, mov_boxes=[], obs_boxes=[wall], obs_pts=np.zeros((0, 3)))


def test_moves_into_an_obstacle_are_rejected():
    geom, center, axis = _geometry(), np.array([0, 0, 0.5]), np.array([0.0, 0, 1])
    assert rm.path_is_clear(geom, center, np.array([-0.03, 0, 0]), axis, 0.0, 0.004, 0.0005) == (True, "ok")
    assert rm.path_is_clear(geom, center, np.array([0.03, 0, 0]), axis, 0.0, 0.004, 0.0005) == (False, "obstacle")
    assert rm.path_is_clear(geom, center, np.array([0, 0, -0.6]), axis, 0.0, 0.004, 0.0005) == (False, "table")


def test_a_point_already_inside_may_leave_but_not_go_deeper():
    geom = _geometry()
    geom["mov_pts"] = geom["mov_pts"] + [0.03, 0, 0]  # the cube's +x face is now inside the grown wall
    center, axis = np.array([0.03, 0, 0.5]), np.array([0.0, 0, 1])
    assert rm.path_is_clear(geom, center, np.array([-0.02, 0, 0]), axis, 0.0, 0.004, 0.0005)[0]  # pull out
    assert not rm.path_is_clear(geom, center, np.array([0.01, 0, 0]), axis, 0.0, 0.004, 0.0005)[0]  # push in
