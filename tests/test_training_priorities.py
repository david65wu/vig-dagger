from pathlib import Path
from graphlib import TopologicalSorter
from vig_dagger.config import load
from vig_dagger.plan import build


def test_requested_two_seed_schedule(tmp_path):
    cfg=load(Path(__file__).parents[1]/'configs/run.yaml',[f'work_dir={tmp_path}'])
    jobs={j.name:j for j in build(cfg,tmp_path/'config.json')}
    list(TopologicalSorter({n:set(j.deps) for n,j in jobs.items()}).static_order())
    assert cfg['retrain']['seeds']==[0,1]
    assert not any('/seed2' in name for name in jobs)
    previous=set();all_prioritized=set()
    for group in cfg['retrain']['priority_groups']:
        current={f'train/retrain/{dataset}/seed{s}' for dataset in group for s in [0,1]}
        assert all(previous <= set(jobs[n].deps) for n in current)
        assert max(jobs[n].priority for n in current if n.endswith('seed0')) < min(jobs[n].priority for n in current if n.endswith('seed1'))
        if previous:assert max(jobs[n].priority for n in previous)<min(jobs[n].priority for n in current)
        previous=current;all_prioritized|=current
    assert len(all_prioritized)==24
    assert not all_prioritized.intersection(jobs['train/retrain/base/seed1'].deps)
    assert jobs['train/retrain/base/seed1'].priority==0
    assert all(len(j.deps)==1 and j.deps[0].startswith('train/') for n,j in jobs.items() if n.startswith('eval/'))
    assert cfg['evaluation']['num_episodes']==100
    assert not any(n.startswith('eval/driver/') for n in jobs)
    assert all(60000<=int(n.rsplit('/',1)[1])<60100 for n in jobs if n.startswith('eval/'))
    assert not any('base_plus' in d for d in jobs['train/retrain/base/seed0'].deps)


def test_blocked_traverses_shared_dependencies_once():
    from types import SimpleNamespace
    from vig_dagger.jobs import Queue, Job
    jobs={'base':Job('base',[])}
    previous=['base']
    for layer in range(8):
        current=[]
        for i in range(8):
            name=f'{layer}_{i}';jobs[name]=Job(name,[],deps=previous);current.append(name)
        previous=current
    end=Job('end',[],deps=previous)
    queue=SimpleNamespace(jobs=jobs,failed=set())
    assert not Queue.blocked(queue,end)
    queue.failed.add('base')
    assert Queue.blocked(queue,end)


def test_moe_only_driver_evaluation_keeps_retraining_schedule(tmp_path):
    cfg = load(Path(__file__).parents[1] / 'configs/run.yaml', [f'work_dir={tmp_path}'])
    cfg['evaluation'].update(evaluate_drivers=True, driver_rounds=['moe'], driver_all_levels=True)
    jobs = {j.name: j for j in build(cfg, tmp_path / 'config.json')}
    driver_evals = [j for n, j in jobs.items() if n.startswith('eval/driver/')]
    assert len(driver_evals) == 28
    assert all(j.name.startswith('eval/driver/moe/') for j in driver_evals)
    assert {int(j.name.split('/')[3][5:]) for j in driver_evals} == {1, 2, 4, 8, 16, 32, 64}
    assert all(j.deps == ['train/drivers/moe'] for j in driver_evals)
    assert sum(n.startswith('eval/retrain/') for n in jobs) == 28 * 4
