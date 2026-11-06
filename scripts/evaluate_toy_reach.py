"""Isaac Lab rollout evaluation or shared static probe capture for toy reaching."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np
import torch


def snapshot(obs):
    return (obs["policy"][0].cpu().numpy().copy(),
            np.stack([obs["rgb_camera"][f"{c}_cam"][0].cpu().numpy().copy() for c in ("third_person", "wrist")]))


def capture(args, env):
    from vig_dagger.toy_reach_env.planner import TubePlan
    records, states, images, references = [], [], [], []
    source = Path('/cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_validation_5_v1/dataset/toy_reach/shards')
    # 100 held-out on-trajectory observations calibrate warning thresholds.
    for path in sorted(source.glob('*.hdf5')):
        with h5py.File(path) as f:
            g = next(iter(f['data'].values()))
            for t in np.linspace(30, len(g['actions']) - 35, 20).astype(int):
                state = g['obs/state'][t]
                rgb = np.stack([g[f'obs/rgb_camera/{c}_cam'][t] for c in ('third_person', 'wrist')])
                states.append(state); images.append(rgb); references.append(g['actions'][t])
                records.append(dict(kind='calibration', source=path.name, frame=int(t), requested_radius=0.0))
    rng = np.random.default_rng(3000)
    a, b = np.asarray(env.cfg.point_a), np.asarray(env.cfg.point_b)
    for progress in (.25, .5, .75):
        angles = rng.uniform(0, 2*np.pi, 8)
        for radius in (0, .005, .01, .02, .03, .05, .08):
            for direction, angle in enumerate(angles if radius else [0]):
                position = a + progress * (b-a) + radius * np.array([np.cos(angle), 0, np.sin(angle)])
                # Reset before each probe so redundant-arm configuration has a consistent initialization.
                obs, _ = env.reset(seed=3000, options={'eef_position':position})
                state, rgb = snapshot(obs)
                reference = env.action_toward(env.cfg.point_b, max_translation=.008)[0].cpu().numpy().copy()
                states.append(state); images.append(rgb); references.append(reference)
                actual_radius = np.linalg.norm((state[:3]-a)[[0,2]])
                records.append(dict(kind='probe', progress=progress, requested_radius=radius,
                                    direction=direction, angle=float(angle), requested_position=position.tolist(),
                                    actual_radius=float(actual_radius), placement_error=float(np.linalg.norm(state[:3]-position))))
            print(f'CAPTURE progress={progress} radius={radius} count={len(records)}', flush=True)
    temporary = args.out/'observations.tmp.hdf5'
    with h5py.File(temporary,'w') as f:
        f.create_dataset('state',data=np.asarray(states))
        f.create_dataset('images',data=np.asarray(images),compression='gzip',compression_opts=1)
        f.create_dataset('reference_action',data=np.asarray(references))
        f.attrs['records']=json.dumps(records)
    temporary.rename(args.out/'observations.hdf5')
    (args.out/'probes_done.json').write_text(json.dumps({'observations':len(records),'calibration':100,'probes':len(records)-100}))


def run_rollouts(args, env):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_evaluation import observation, video
    from vig_dagger.toy_reach_env.planner import TubePlan
    policy = LibraryPolicy(args.run_dir/'latest.pt', 'cuda:0')
    cases=[]
    for seed in range(2000,2010):
        plan=TubePlan.sample(env.cfg.point_a,env.cfg.point_b,.008,seed)
        cases.append(dict(kind='id',seed=seed,start=plan.control_points[0].tolist()))
    for axis in (0,2):
        for radius in (.03,.05):
            for sign in (-1,1):
                point=(np.asarray(env.cfg.point_a)+np.asarray(env.cfg.point_b))/2
                point[axis]+=sign*radius
                cases.append(dict(kind='ood',seed=4000+len(cases),start=point.tolist(),axis=axis,offset=sign*radius))
    results=[]
    for index,case in enumerate(cases):
        obs,_=env.reset(seed=case['seed'],options={'eef_position':case['start']})
        torch.manual_seed(5000+index)
        states,actions,frames=[],[],[]
        stable=0; reason='time_limit'; chunk=None
        for step in range(1200):
            state,rgb=snapshot(obs)
            states.append(state)
            if step%2==0 and (index<2 or case['kind']=='ood'):
                frames.append((rgb[0],rgb[1],f"{args.run_dir.name} | {case['kind']} {index+1} step {step}"))
            if step%8==0:
                s,im=observation(obs,policy.device)
                chunk=policy.act(s,im,factor=1)[0]
            action=chunk[step%8].reshape(1,7).to(env.device)
            if not torch.isfinite(action).all():
                reason='nonfinite_action'; break
            # Abort extreme predictions rather than altering/clipping policy outputs.
            if action[0,:3].norm()>.05 or action[0,3:6].norm()>.2:
                reason='extreme_action'; break
            actions.append(action[0].cpu().numpy().copy())
            obs,_=env.step(action)
            position=obs['policy'][0,:3].cpu().numpy()
            error=float(np.linalg.norm(position-np.asarray(env.cfg.point_b)))
            stable=stable+1 if error<.005 else 0
            if stable>=15:
                reason='success'; break
            if not (.15<position[0]<.75 and -.4<position[1]<.45 and .12<position[2]<.65):
                reason='workspace_exit'; break
        final,rgb=snapshot(obs)
        error=float(np.linalg.norm(final[:3]-np.asarray(env.cfg.point_b)))
        if frames:
            frames.extend([(rgb[0],rgb[1],f"{args.run_dir.name} | {reason} | error {error*1000:.1f} mm")]*15)
            video(args.out/'videos'/f"{index:02d}_{case['kind']}.mp4",frames)
        np.savez_compressed(args.out/f'rollout_{index:02d}.npz',states=np.asarray(states),actions=np.asarray(actions),final_state=final)
        result=dict(case,success=reason=='success',reason=reason,steps=len(actions),goal_error_m=error)
        results.append(result)
        (args.out/'rollouts.json').write_text(json.dumps(results,indent=2))
        print('ROLLOUT '+json.dumps(result),flush=True)
    (args.out/'rollouts_done.json').write_text(json.dumps({'episodes':len(results)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['probes','rollouts'],required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run-dir',type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    class Launcher(AppLauncher):
        def _resolve_device_settings(self,values):
            super()._resolve_device_settings(values)
            values['active_gpu']=args.gpu; values['multi_gpu']=False
    app=Launcher(dict(headless=True,enable_cameras=True,device='cpu')).app
    try:
        from vig_dagger.toy_reach_env.config import ToyReachEnvCfg
        from vig_dagger.toy_reach_env.environment import ToyReachEnv
        with torch.inference_mode():
            env=ToyReachEnv(ToyReachEnvCfg())
            try:
                (capture if args.mode=='probes' else run_rollouts)(args,env)
            finally:
                env.close()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    app.close()
