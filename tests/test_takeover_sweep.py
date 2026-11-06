"""The single-machine sweep must isolate collections and allow independent retraining."""
import json
from pathlib import Path

from vig_dagger.config import Paths, load, scorer_type, training_sets
from vig_dagger.plan import build
from vig_dagger.steps import dagger_settings


def test_sweep_graph(tmp_path):
    cfg = load(Path(__file__).parents[1] / 'configs/run.yaml', [f'work_dir={tmp_path}'])
    cfg['retrain']['prioritize_shared'] = False
    cfg['retrain'].pop('priority_groups', None)  # Verify the default overlapping schedule independently.
    jobs = build(cfg, tmp_path / 'config.json')
    by_name = {j.name: j for j in jobs}
    assert len(jobs) == len(by_name)
    assert all(d in by_name for j in jobs for d in j.deps)
    drivers = {j.name for j in jobs if j.name.startswith('train/drivers/')}
    drivers.add('train/retrain/base/seed0')
    assert len(drivers) == 8
    for name in drivers:
        assert not any('base_plus_' in d for d in by_name[name].deps)
        assert by_name[name].priority == 10
    datasets = training_sets(cfg)
    assert len(datasets) == len(set(datasets)) == 14
    for r in cfg['rounds']:
        for variant in cfg['dagger']['scorers']:
            prepare = by_name[f'dagger_prepare_{r}_{variant}']
            assert drivers <= set(prepare.deps)
            workers = [j for j in jobs if j.name.startswith(f'dagger_{r}_{variant}_')]
            assert len(workers) == 4
            for seed in cfg['retrain']['seeds']:
                name = f'train/retrain/base_plus_{r}_{variant}/seed{seed}'
                assert by_name[name].deps == [f'combine_{r}_{variant}']
                assert by_name[name].priority < workers[0].priority
    # Prove the dependency graph is acyclic.
    remaining = set(by_name)
    done = set()
    while remaining:
        ready = {n for n in remaining if set(by_name[n].deps) <= done}
        assert ready
        remaining -= ready
        done |= ready
    assert set(n for n in by_name if n.startswith('analyze_')) <= set(by_name['summarize'].deps)


def test_variant_thresholds_and_paths(tmp_path):
    cfg = load(Path(__file__).parents[1] / 'configs/run.yaml', [f'work_dir={tmp_path}'])
    paths = Paths.of(cfg)
    outputs = set()
    for r in cfg['rounds']:
        paths.thresholds(r).parent.mkdir(parents=True, exist_ok=True)
        paths.thresholds(r).write_text(json.dumps({s: {'p50': .5, 'p90': .9, 'p95': .95, 'p99': .99} for s in ['vig','flow']}))
        for v in cfg['dagger']['scorers']:
            settings = dagger_settings(cfg, r, v)
            assert settings['scorer_type'] == scorer_type(cfg, v) == v.split('_')[0]
            assert settings['take_threshold'] == int(v.split('_p')[1]) / 100
            assert settings['return_threshold'] == .5
            assert settings['target'] == 100
            outputs.add(paths.dagger(r, v))
    assert len(outputs) == 12


def test_collection_gpu_placement_spreads_across_eight_devices(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    from vig_dagger.jobs import Job, Queue, Running

    cfg = load(Path(__file__).parents[1] / 'configs/run.yaml', [f'work_dir={tmp_path}'])
    queue = Queue([], cfg['resources'], tmp_path / 'logs')
    monkeypatch.setattr(queue.machine, 'gpu_memory', lambda: {g: (48., 0.) for g in range(8)})
    monkeypatch.setattr('vig_dagger.jobs.psutil.virtual_memory', lambda: SimpleNamespace(available=500 * 2**30))
    monkeypatch.setattr('vig_dagger.jobs.os.getloadavg', lambda: (0., 0., 0.))
    placed = []
    for i in range(8):
        job = Job(f'collection{i}', ['vig_dagger.workers.dagger'], kind='sim')
        gpu = queue.place(job)
        assert gpu is not False
        placed.append(gpu)
        queue.running[job.name] = Running(job, 1000 + i, gpu, time.time())
    assert set(placed) == set(range(8))


def test_renderer_uses_assigned_physical_gpu_with_cpu_physics(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    from vig_dagger import isaac

    captured = {}
    class FakeLauncher:
        def __init__(self, args):
            captured.update(args)
            self._resolve_device_settings(captured)
            self.app = 'simulator'
        def _resolve_device_settings(self, args):
            args.update(active_gpu=0, physics_gpu=0)
    fake = ModuleType('isaaclab.app')
    fake.AppLauncher = FakeLauncher
    monkeypatch.setitem(sys.modules, 'isaaclab.app', fake)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr('vig_dagger.jobs.exit_on_sigterm', lambda: None)
    assert isaac.launch() == 'simulator'
    assert captured['active_gpu'] == 7
    assert captured['multi_gpu'] is False
    assert captured['device'] == 'cpu'


def test_threshold_merge_closes_shards_before_cleanup(tmp_path, monkeypatch):
    import numpy as np
    import shutil
    from vig_dagger import steps
    from vig_dagger.workers.score_demos import part_path

    cfg = load(Path(__file__).parents[1] / 'configs/run.yaml', [f'work_dir={tmp_path}', 'thresholds.chunks=2'])
    for c in range(2):
        p = part_path(cfg, 'moe', c)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, vig=np.array([c + 1.]), flow=np.array([c + 2.]), vig_gap=np.array([[c + 1.]]),
                 demo=np.array([c]), frame=np.array([0]), factors=np.array([1, 2]))
    opened = []
    original_load, original_cleanup = np.load, shutil.rmtree
    def track_load(*args, **kwargs):
        part = original_load(*args, **kwargs)
        opened.append(part)
        return part
    def require_closed(path):
        assert opened and all(part.zip is None for part in opened)
        original_cleanup(path)
    monkeypatch.setattr(steps.np, 'load', track_load)
    monkeypatch.setattr(steps.shutil, 'rmtree', require_closed)
    steps.thresholds(cfg, 'moe')
    table = json.loads(Paths.of(cfg).thresholds('moe').read_text())
    assert table['frames'] == 2
    assert table['vig']['p50'] == 1.5
    assert not part_path(cfg, 'moe', 0).parent.exists()
