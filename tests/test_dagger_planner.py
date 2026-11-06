"""Grasp feasibility and control-flow regressions, using Isaac Lab's real quaternion math."""
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from vig_dagger.expert.grasp_search import grasp_yaws, wrap


def test_search_finds_angle_missed_by_nearest_only_selector():
    # Saved state near the start of 1000096's charger approach: nearest yaw
    # predicts q7 beyond its upper limit, but another allowed yaw has room.
    current, ideal, swing = map(math.radians, (-82.66, -72.84, 44.99))
    q7, limits = 2.246, (-2.8973, 2.8973)
    assert q7 + swing > limits[1]
    options = grasp_yaws([ideal, ideal + math.pi], current, q7, limits, swing, -swing)
    assert options
    assert abs(wrap(options[0].yaw - ideal)) <= math.radians(25) + 1e-9
    assert all(o.margin >= .3 for o in options)
    assert options[0].yaw > current


def test_no_infeasible_fallback_and_retry_excludes_previous_angle():
    assert not grasp_yaws([0], 0, 10, (-1, 1), 0, 0)
    first = grasp_yaws([0, math.pi], 0, 0, (-2.9, 2.9), 0, 0)[0]
    other = grasp_yaws([0, math.pi], 0, 0, (-2.9, 2.9), 0, 0, excluded=[first.yaw])
    assert other and all(abs(wrap(o.yaw - first.yaw)) >= math.radians(10) - 1e-9 for o in other)


def test_revision_guard_prevents_mixing_old_and_new_collections(tmp_path):
    import json
    from vig_dagger.config import Paths, load
    from vig_dagger.steps import dagger_prepare, dagger_settings
    cfg = load(Path(__file__).parents[1] / 'configs/run.yaml', [f'work_dir={tmp_path}'])
    paths = Paths.of(cfg)
    paths.thresholds('moe').parent.mkdir(parents=True, exist_ok=True)
    paths.thresholds('moe').write_text(json.dumps({'vig': {'p99': .99, 'p50': .5}}))
    old = dagger_settings(cfg, 'moe', 'vig_p99')
    old.pop('planner_revision')
    out = paths.dagger('moe', 'vig_p99')
    out.mkdir(parents=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(old))
    before = settings.read_bytes()
    with pytest.raises(SystemExit, match='planner_revision'):
        dagger_prepare(cfg, 'moe', 'vig_p99')
    assert settings.read_bytes() == before
    cfg['dagger']['planner_revision'] = 'legacy'
    dagger_prepare(cfg, 'moe', 'vig_p99')
    assert json.loads(settings.read_text())['planner_revision'] == 'legacy'


@pytest.fixture
def modules(monkeypatch):
    # The math module itself needs no Kit; use the installed source, not mock math.
    path = Path(__file__).parents[1] / 'third_party/IsaacLab/source/isaaclab/isaaclab/utils/math.py'
    if not path.exists():
        pytest.skip('Isaac Lab source is unavailable')
    spec = importlib.util.spec_from_file_location('isaaclab.utils.math', path)
    maths = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'isaaclab', ModuleType('isaaclab'))
    monkeypatch.setitem(sys.modules, 'isaaclab.utils', ModuleType('isaaclab.utils'))
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.math', maths)
    spec.loader.exec_module(maths)
    from vig_dagger.expert import planner as base, dagger_planner as new
    yield base, new
    # These imports depend on the temporary package entries above.
    for name in ('vig_dagger.expert.planner', 'vig_dagger.expert.dagger_planner'):
        sys.modules.pop(name, None)
    import vig_dagger.expert as package
    for name in ('planner', 'dagger_planner'):
        if hasattr(package, name):
            delattr(package, name)


def bare(cls):
    p = cls.__new__(cls)
    p.device = 'cpu'
    p.speed_transit = p.speed_careful = .1
    p.steps = 0
    p.events = []
    p.trace_enabled = False
    p._q7_limits = (-2.8973, 2.8973)
    p.q7 = lambda: 2.8973
    p.tcp_pose = lambda: (torch.zeros(3), torch.tensor([1., 0., 0., 0.]))
    def action(dp, dr, grip, speed):
        p.steps += 1
        return torch.cat((dp, dr, torch.tensor([grip])))
    p._action = action
    return p


def exhaust(generator):
    actions = []
    while True:
        try:
            actions.append(next(generator))
        except StopIteration as end:
            return actions, end.value


def test_legacy_freeze_preserved_but_dagger_requires_real_alignment(modules):
    base, new = modules
    target = base.quat_from_euler_xyz(torch.tensor([.4]), torch.zeros(1), torch.zeros(1))[0]
    old = bare(base.OpenDrawerPlanner)
    _, reached = exhaust(old.servo(torch.zeros(3), target, base.OPEN))
    assert reached  # Historical pure-MP behavior intentionally unchanged.
    expert = bare(new.DaggerPlanner)
    with pytest.raises(new.ApproachFailed, match='rotation stalled'):
        exhaust(expert.servo(torch.zeros(3), target, base.OPEN, name='charger descend'))
    assert expert.steps <= 60


def test_recovery_is_bounded_and_only_retries_pregrasp_failures(modules, monkeypatch):
    base, new = modules
    attempts = []
    def fail(self):
        attempts.append(1)
        raise new.ApproachFailed('charger descend: test stall')
        yield
    def clear(*args, **kwargs):
        return True
        yield
    monkeypatch.setattr(base.OpenDrawerPlanner, '_pick_charger', fail)
    expert = bare(new.DaggerPlanner)
    expert.charger_center = lambda: torch.zeros(3)
    expert._approach_servo = clear
    with pytest.raises(base.PlanFailed, match='recovery exhausted'):
        exhaust(expert._pick_charger())
    assert len(attempts) == 3
    assert sum(e['kind'] == 'approach_retry' for e in expert.events) == 2
    def lost_grasp(self):
        raise base.PlanFailed('charger dropped during the lift')
        yield
    monkeypatch.setattr(base.OpenDrawerPlanner, '_pick_charger', lost_grasp)
    expert.events.clear()
    with pytest.raises(base.PlanFailed, match='dropped during the lift'):
        exhaust(expert._pick_charger())
    assert not expert.events
