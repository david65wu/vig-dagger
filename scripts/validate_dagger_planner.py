"""Replay saved actions up to a takeover, then compare continuous expert execution.

Read-only with respect to source collections. Outputs go to a separate directory.
This tests expert continuation, not DAgger success under a live switching gate.
Run with .venv-isaac, PYTHONHASHSEED=0 and the configured Isaac environment.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from vig_dagger import isaac


def body(args):
    from vig_dagger.expert import planner as P
    from vig_dagger.expert.dagger_planner import DaggerPlanner

    P.set_expert_mode('demo')
    env, success_term = isaac.make_env()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for path in (args.collection / 'episodes').glob('*.jsonl')
            for line in path.read_text().splitlines() if line.strip()]
    results = []
    try:
        for seed in args.seeds:
            row = next(r for r in rows if r['seed'] == seed)
            with h5py.File(args.collection / row['file']) as f:
                g = f[f"data/demo_{row['traj']}"]
                actions = g['actions'][:]
                start = int(g['dagger/take_steps'][-1])
                strategy = json.loads(g.attrs['expert_strategies'])[-1]
                expected = {k: g[f'task_state/{k}'][start] for k in ('tcp_pose', 'qpos', 'charger_pose')}
                def read_state(group):
                    return {k: read_state(v) if isinstance(v, h5py.Group) else torch.tensor(v[:])
                            for k, v in group.items()}
                initial = read_state(g['initial_state'])
            for revision in args.revisions:
                torch.manual_seed(seed)
                np.random.seed(seed % 2**32)
                isaac.reset(env, seed)
                env.reset_to(initial, env_ids=torch.tensor([0], device=env.device), seed=seed, is_relative=True)
                rule = isaac.SuccessRule(env, success_term)
                for action in actions[:start]:
                    env.step(torch.tensor(action, dtype=torch.float32).reshape(1, -1))
                    rule.update()
                params = P.StrategyParams(**strategy)
                plan = (P.OpenDrawerPlanner(env, params, expert=True, trace=True) if revision == 'legacy'
                        else DaggerPlanner(env, params, trace=True))
                log = isaac.TaskStateLog()
                log.reset(env)
                actual = log.arrays()
                mismatch = {k: float(np.max(np.abs(actual[k][0] - expected[k]))) for k in expected}
                # Quaternion signs can differ while representing the same rotation.
                mismatch['tcp_position_m'] = float(np.linalg.norm(actual['tcp_pose'][0, :3] - expected['tcp_pose'][:3]))
                equivalent = (mismatch['qpos'] < .03 and mismatch['tcp_position_m'] < .005
                              and np.linalg.norm(actual['charger_pose'][0, :3] - expected['charger_pose'][:3]) < .005)
                print(f'REPLAY seed={seed} revision={revision} start={start} match={equivalent} {mismatch}', flush=True)
                reason, success, count = 'time_limit_expert', False, start
                for action in plan.actions():
                    if count >= args.max_steps:
                        break
                    env.step(action.reshape(1, -1))
                    count += 1
                    status = rule.update()
                    log.step(env, 1)
                    if status == 'success':
                        reason, success = 'success_expert', True
                        break
                else:
                    reason = plan.failure or 'expert_ended'
                result = dict(seed=seed, revision=revision, start=start, steps=count, success=success,
                              reason=reason, replay_matches=bool(equivalent), replay_error=mismatch,
                              events=getattr(plan, 'events', []),
                              latches=isaac.latches(env))
                results.append(result)
                np.savez_compressed(args.out / f'{seed}_{revision}.npz', **log.arrays())
                (args.out / f'{seed}_{revision}.json').write_text(json.dumps(result, indent=2))
                (args.out / 'results.json').write_text(json.dumps(results, indent=2))
                print('RESULT ' + json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[1000096])
    p.add_argument('--revisions', nargs='+', choices=['legacy', 'dagger_recovery_v1'],
                   default=['legacy', 'dagger_recovery_v1'])
    p.add_argument('--max-steps', type=int, default=1200)
    args = p.parse_args()
    if args.out.exists() and any(args.out.glob('*_legacy.json')):
        raise SystemExit('Use a fresh validation output directory.')
    with torch.inference_mode():
        isaac.run(body, args)
