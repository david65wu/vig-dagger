"""Continue the six sphere policies and measure success on their own training pairs."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from vig_dagger.toy_training import launch, write_json

REPO = Path(__file__).resolve().parents[1]
EPOCHS = (100, 150, 200, 250, 300)


def prepare(root, previous):
    import h5py
    from robopolicy import RunConfig
    original = json.loads((previous / 'experiment.json').read_text())
    runs = []
    for old in original['runs']:
        spec = copy.deepcopy(old)
        out = root / 'policies' / spec['name']
        out.mkdir(parents=True)
        source = previous / 'policies' / spec['name']
        config = json.loads((source / 'run_config.json').read_text())
        config['logging']['out_dir'] = str(out)
        config['training']['max_updates'] = 200 * spec['updates_per_epoch']
        spec.update(run_dir=str(out), epochs=300, initial_epochs=100, additional_epochs=200,
                    total_updates=config['training']['max_updates'],
                    initialize_from=str(source / 'epoch-100.pt'),
                    checkpoints={str((e - 100) * spec['updates_per_epoch']): f'epoch-{e:03d}.pt'
                                 for e in EPOCHS[1:]})
        write_json(out / 'run_config.json', RunConfig.model_validate(config).model_dump(mode='json'))
        write_json(out / 'experiment.json', spec)
        (out / 'epoch-100.pt').symlink_to(source / 'epoch-100.pt')
        runs.append(spec)
    cases = []
    source = Path(original['source']) / 'toy_reach/shards'
    for i in range(20):
        with h5py.File(source / f'shard_{i:05d}.hdf5') as f:
            demo = next(iter(f['data'].values()))
            a, b = demo.attrs['point_a'].tolist(), demo.attrs['point_b'].tolist()
            cases.append(dict(case_id=i, kind='familiar_a_to_b', demo=i, point_a=a, point_b=b,
                              start=a, reset_seed=int(demo.attrs['seed']), policy_seed=81015+i))
    for epoch in EPOCHS:
        directory = root / 'eval' / f'epoch-{epoch:03d}'
        directory.mkdir(parents=True)
        write_json(directory / 'cases.json', cases)
    write_json(root / 'experiment.json', dict(
        runs=runs, previous_root=str(previous), created_at=time.time(), source=str(source),
        continuation='Epoch-100 weights; fresh AdamW; 100-update warmup to 1e-4; cosine decay over 200 more epochs',
        unchanged='Data selection, normalization, architecture, augmentation, batch size, action semantics',
        evaluation=dict(epochs=EPOCHS, own_training_pairs=True, trials_per_pair=1,
                        total_rollouts=375, image_factor=1, horizon=16, execute_steps=8,
                        success='within 5 mm of B for 15 consecutive steps', max_steps=1200,
                        policy_seeds='81015 + demo index, identical across checkpoints and policies'),
        epoch_convention='Checkpoint names include original 100 epochs; optimizer steps count only this continuation'))
    for relative in ['scripts/continue_sphere_training.py', 'scripts/rollout_sphere_simple.py',
                     'scripts/toy_flow_search_sim.py', 'src/vig_dagger/toy_training.py',
                     'src/robopolicy/training/trainer.py', 'src/vig_dagger/policy.py',
                     'src/vig_dagger/toy_reach_env/environment.py',
                     'src/vig_dagger/toy_reach_env/sphere_environment.py']:
        dest = root / 'source_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, dest)


def report(root, specs):
    rows = []
    for spec in specs:
        for epoch in EPOCHS:
            out = root / 'eval' / f'epoch-{epoch:03d}' / spec['name']
            status_path = out / 'status.json'
            if not status_path.exists() or json.loads(status_path.read_text())['state'] != 'complete':
                continue
            results = json.loads((out / 'rollouts.json').read_text())
            assert len(results) == spec['num_demos']
            assert {r['demo'] for r in results} == set(range(spec['num_demos']))
            success = sum(r['success'] for r in results)
            rows.append(dict(policy=spec['name'], epoch=epoch, successes=success,
                             episodes=len(results), success_rate=success / len(results)))
    write_json(root / 'trained_pair_success.json', rows)
    if rows:
        with (root / 'trained_pair_success.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for spec in specs:
            selected = [r for r in rows if r['policy'] == spec['name']]
            if selected:
                ax.plot([r['epoch'] for r in selected], [r['success_rate'] * 100 for r in selected],
                        'o-', label=spec['name'].replace('_', ' '))
        ax.set(xlabel='Total training epochs', ylabel='Success on own training A/B pairs (%)',
               xticks=EPOCHS, ylim=(-2, 102), title='One rollout per training pair, fixed seeds across checkpoints')
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(root / 'trained_pair_success.png', dpi=160); plt.close(fig)
    return len(rows)


def run(root, previous):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / 'experiment.json').exists():prepare(root, previous)
        manifest = json.loads((root / 'experiment.json').read_text())
        specs = manifest['runs']
        launch(argparse.Namespace(root=root))
        setup = root.parent / 'setup'
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', OMNI_KIT_ALLOW_ROOT='1',
                   PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
                   VK_ICD_FILENAMES=str(setup/'nvidia-570/nvidia_icd_local.json'),
                   __EGL_VENDOR_LIBRARY_FILENAMES=str(setup/'nvidia-570/10_nvidia_local.json'))
        env['LD_LIBRARY_PATH'] = str(setup/'nvidia-graphics') + ':' + env.get('LD_LIBRARY_PATH', '')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        active = {}; completed = set(); started = time.time()
        while True:
            states = {}
            for spec in specs:
                path = Path(spec['run_dir']) / 'status.json'
                states[spec['name']] = json.loads(path.read_text()) if path.exists() else {'state':'starting'}
                if states[spec['name']]['state'] == 'failed':raise RuntimeError(states[spec['name']])
            for key, job in list(active.items()):
                if job['process'].poll() is not None:
                    status_path = job['out'] / 'status.json'
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                    if job['process'].returncode or status.get('state') != 'complete':
                        raise RuntimeError(f'Evaluation {key} failed: {status}')
                    completed.add(key); del active[key]; report(root, specs)
            free = {6, 7} | {s['gpu'] for s in specs if states[s['name']]['state'] == 'complete'}
            free -= {job['gpu'] for job in active.values()}
            for epoch in EPOCHS:
                for spec in specs:
                    name = spec['name']; key = (epoch, name)
                    out = root / 'eval' / f'epoch-{epoch:03d}' / name
                    status_path = out / 'status.json'
                    if status_path.exists() and json.loads(status_path.read_text())['state'] == 'complete':
                        completed.add(key)
                    if key in completed or key in active or not free:continue
                    if not (Path(spec['run_dir']) / f'epoch-{epoch:03d}.pt').exists():continue
                    gpu = min(free); free.remove(gpu)
                    cmd = [str(REPO/'.venv-isaac/bin/python'), str(REPO/'scripts/rollout_sphere_simple.py'),
                           '--root', str(out.parent), '--name', name, '--gpu', str(gpu),
                           '--training', str(root), '--checkpoint', f'epoch-{epoch:03d}.pt',
                           '--case-count', str(spec['num_demos']), '--source', manifest['source']]
                    with (root/'logs'/f'eval_epoch{epoch}_{name}.log').open('a') as log:
                        process = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = dict(process=process, gpu=gpu, out=out)
                    print(f'EVAL {key} GPU {gpu} PID {process.pid}', flush=True)
            finished = len(completed) == 30 and all(s['state'] == 'complete' for s in states.values())
            write_json(root/'pipeline.json', dict(state='complete' if finished else 'running',
                pid=os.getpid(), updated_at=time.time(), elapsed_seconds=time.time()-started,
                training=states, evaluations_complete=len(completed), evaluations_total=30,
                active_evaluations=[dict(epoch=k[0], policy=k[1], gpu=v['gpu'], pid=v['process'].pid)
                                    for k,v in active.items()]))
            if finished:report(root,specs);break
            time.sleep(10)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    args=p.parse_args()
    try:run(args.root.resolve(), args.previous.resolve())
    except BaseException as exc:
        write_json(args.root/'pipeline.json', dict(state='failed', error=repr(exc), pid=os.getpid()))
        traceback.print_exc();raise
