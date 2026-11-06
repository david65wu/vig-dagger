"""Closed-loop sphere success rates for familiar, unseen, and plotted ID/OOD starts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np
import torch


def write(path, data):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def reset_case(env, case, source):
    from toy_flow_search_sim import read_tree, restore, snapshot
    # Evaluation-only larger sphere; training configuration stays unchanged.
    env.cfg.sphere_radius=case.get('sphere_radius',.15)
    obs,_=env.reset(seed=case['reset_seed'],options=dict(point_a=case['point_a'],point_b=case['point_b'],
                                                       eef_position=case['start']))
    check={}
    if case['kind']=='midpoint_id':
        # Reconstruct the recorded physical frame, including joint velocities.
        with h5py.File(source/f"shard_{case['demo']:05d}.hdf5") as f:
            g=next(iter(f['data'].values()));frame=case['frame']
            scene=read_tree(g['initial_state'],env.device)
            robot=scene['articulation']['robot']
            robot['joint_position']=torch.tensor(g['obs/joint_pos'][frame],device=env.device)[None]
            robot['joint_velocity']=torch.tensor(g['obs/joint_vel'][frame],device=env.device)[None]
            obs=restore(env,scene)
            state,rgb=snapshot(obs)
            recorded=g['obs/state'][frame]
            check['recorded_state_max_error']=float(np.abs(state-recorded).max())
            check['recorded_position_error_m']=float(np.linalg.norm(state[:3]-recorded[:3]))
            assert check['recorded_state_max_error']<1e-4,check
    state,rgb=snapshot(obs)
    assert np.allclose(state[9:12],case['point_b'],atol=1e-7)
    check['placement_error_m']=float(np.linalg.norm(state[:3]-case['start']))
    assert check['placement_error_m']<.001,check
    return obs,check


def run(args):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_evaluation import observation,video
    from vig_dagger.toy_reach_env.sphere_config import SphereReachEnvCfg
    from vig_dagger.toy_reach_env.sphere_environment import SphereReachEnv
    from toy_flow_search_sim import snapshot
    out=args.root/args.name;out.mkdir(exist_ok=True)
    cases=json.loads((args.root/'cases.json').read_text())
    if args.case_count is not None:cases=cases[:args.case_count]
    policy=LibraryPolicy(args.training/'policies'/args.name/args.checkpoint,args.policy_device or f'cuda:{args.gpu}')
    env=SphereReachEnv(SphereReachEnvCfg())
    results=json.loads((out/'rollouts.json').read_text()) if (out/'rollouts.json').exists() else []
    done={r['case_id'] for r in results};started=time.time()
    write(out/'status.json',dict(state='running',pid=os.getpid(),gpu=args.gpu,completed=len(results),total=len(cases)))
    try:
        for case in cases:
            if case['case_id'] in done:continue
            obs,checks=reset_case(env,case,args.source)
            initial_state,initial_rgb=snapshot(obs)
            torch.manual_seed(case['policy_seed'])
            states,actions,frames=[],[],[]
            stable=0;reason='time_limit'
            for step in range(1200):
                state=obs['policy'][0].cpu().numpy().copy();states.append(state)
                if step%4==0:
                    _,rgb=snapshot(obs)
                    frames.append((rgb[0],rgb[1],f"{args.name} | {case['kind']} #{case['case_id']} | {step/60:.1f}s"))
                if step%8==0:
                    s,im=observation(obs,policy.device);chunk=policy.act(s,im,factor=1)[0]
                action=chunk[step%8].reshape(1,7).to(env.device)
                if not torch.isfinite(action).all():reason='nonfinite_action';break
                if action[0,:3].norm()>.05 or action[0,3:6].norm()>.2:reason='extreme_action';break
                actions.append(action[0].cpu().numpy().copy());obs,_=env.step(action)
                position=obs['policy'][0,:3].cpu().numpy()
                error=float(np.linalg.norm(position-np.array(case['point_b'])))
                stable=stable+1 if error<.005 else 0
                if stable>=15:reason='success';break
                if not (.15<position[0]<.75 and -.4<position[1]<.45 and .12<position[2]<.65):
                    reason='workspace_exit';break
            final,rgb=snapshot(obs)
            error=float(np.linalg.norm(final[:3]-case['point_b']))
            stem=f"{case['case_id']:02d}_{case['kind']}"
            frames.extend([(rgb[0],rgb[1],f'{args.name} | {reason} | goal error {error*1000:.1f} mm')]*8)
            video(out/'videos'/f'{stem}.mp4',frames,fps=15)
            np.savez_compressed(out/f'{stem}.npz',states=np.asarray(states),actions=np.asarray(actions).reshape(-1,7),
                                final_state=final,initial_state=initial_state,initial_images=initial_rgb)
            result=dict(case,success=reason=='success',reason=reason,steps=len(actions),goal_error_m=error,
                        **checks,video=f'videos/{stem}.mp4')
            results.append(result);write(out/'rollouts.json',results)
            write(out/'status.json',dict(state='running',pid=os.getpid(),gpu=args.gpu,completed=len(results),
                                        total=len(cases),elapsed_seconds=time.time()-started))
            print('ROLLOUT '+json.dumps(result),flush=True)
        write(out/'status.json',dict(state='complete',pid=os.getpid(),gpu=args.gpu,completed=len(results),total=len(cases),
                                    elapsed_seconds=time.time()-started))
        write(out/'completed.json',dict(episodes=len(results),checkpoint=str(args.training/'policies'/args.name/args.checkpoint)))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--checkpoint',default='epoch-100.pt')
    p.add_argument('--case-count',type=int)
    p.add_argument('--policy-device',help='Optional CUDA device override when CUDA_VISIBLE_DEVICES masks the physical GPU')
    p.add_argument('--training',type=Path,default=Path('/cpfs02/user/caojianuo/vig-dagger-runs/toy_sphere_training_v1'))
    p.add_argument('--source',type=Path,default=Path('/cpfs02/user/caojianuo/vig-dagger-runs/toy_sphere_20_demos_v1/dataset/toy_reach/shards'))
    args=p.parse_args()
    from isaaclab.app import AppLauncher
    class Launcher(AppLauncher):
        def _resolve_device_settings(self,values):
            super()._resolve_device_settings(values);values['active_gpu']=args.gpu;values['multi_gpu']=False
    app=Launcher(dict(headless=True,enable_cameras=True,device='cpu')).app
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.inference_mode():run(args)
    except BaseException as exc:
        out=args.root/args.name;out.mkdir(exist_ok=True)
        write(out/'status.json',dict(state='failed',error=repr(exc),pid=os.getpid()))
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    app.close()
