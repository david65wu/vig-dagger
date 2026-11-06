"""Capture matched on-path and off-path sphere observations, without rollouts."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np
import torch


def capture(args):
    from vig_dagger.toy_reach_env.sphere_config import SphereReachEnvCfg
    from vig_dagger.toy_reach_env.sphere_environment import SphereReachEnv
    env = SphereReachEnv(SphereReachEnvCfg())
    records, states, images, references = [], [], [], []
    paths = sorted((args.source/'dataset/toy_reach/shards').glob('*.hdf5'))[:5]
    assert len(paths) == 5
    def append(state, rgb, record, reference):
        assert state.shape == (12,) and np.isfinite(state).all()
        assert rgb.shape == (2, 256, 256, 3) and rgb.dtype == np.uint8
        assert all(im.std() > 2 for im in rgb)
        states.append(state); images.append(rgb); records.append(record); references.append(reference)
    try:
        for demo, path in enumerate(paths):
            with h5py.File(path) as f:
                g = next(iter(f['data'].values()))
                a, b = np.array(g.attrs['point_a']), np.array(g.attrs['point_b'])
                trajectory = g['obs/state'][:, :3]
                along = b-a; along /= np.linalg.norm(along)
                axis = np.eye(3)[np.argmin(np.abs(along))]
                u = np.cross(along, axis); u /= np.linalg.norm(u)
                v = np.cross(along, u)
                progress = (trajectory-a) @ (b-a) / np.dot(b-a, b-a)
                for fraction in (.25, .5, .75):
                    frame = int(np.argmin(np.abs(progress-fraction)))
                    anchor = trajectory[frame]
                    common = dict(demo=demo, frame=frame, progress=fraction,
                                  point_a=a.tolist(), point_b=b.tolist(), anchor=anchor.tolist(),
                                  along=along.tolist(), transverse_u=u.tolist(), transverse_v=v.tolist())
                    recorded_state = g['obs/state'][frame]
                    recorded_rgb = np.stack([g[f'obs/rgb_camera/{c}_cam'][frame] for c in ('third_person','wrist')])
                    append(recorded_state, recorded_rgb, dict(common, kind='recorded_id', radius_m=0., direction=0),
                           g['actions'][frame, :3])
                    cases = [(0., 0, np.zeros(3))]
                    cases += [(r, d, r*vec) for r in (.03,.08) for d,vec in enumerate((u,-u,v,-v))]
                    for radius, direction, offset in cases:
                        requested = anchor+offset
                        obs, _ = env.reset(seed=20000+demo, options=dict(point_a=a,point_b=b,eef_position=requested))
                        state = obs['policy'][0].cpu().numpy().copy()
                        rgb = np.stack([obs['rgb_camera'][f'{c}_cam'][0].cpu().numpy().copy()
                                        for c in ('third_person','wrist')])
                        error = float(np.linalg.norm(state[:3]-requested))
                        nearest = float(np.linalg.norm(trajectory-state[:3],axis=1).min())
                        assert error < .001 and np.allclose(state[9:12], b, atol=1e-7)
                        if radius: assert nearest > radius-.012, (radius, nearest)
                        reference = env.action_toward(b, max_translation=.008)[0,:3].cpu().numpy().copy()
                        append(state, rgb, dict(common,kind='id' if radius==0 else 'ood',radius_m=radius,
                               direction=direction,placement_error_m=error,nearest_same_goal_demo_m=nearest,
                               requested_position=requested.tolist()), reference)
                    (args.root/'capture_progress.json').write_text(json.dumps(dict(observations=len(records),total=150)))
                    print(f'CAPTURE demo={demo} progress={fraction} observations={len(records)}/150',flush=True)
        with h5py.File(args.root/'observations.tmp.hdf5','w') as f:
            f.create_dataset('states',data=np.array(states))
            f.create_dataset('images',data=np.array(images),compression='gzip',compression_opts=1)
            f.create_dataset('references',data=np.array(references))
            f.attrs['records']=json.dumps(records)
        (args.root/'observations.tmp.hdf5').replace(args.root/'observations.hdf5')
        (args.root/'capture_done.json').write_text(json.dumps(dict(observations=len(records),id=15,ood_3cm=60,ood_8cm=60,recorded_id_checks=15)))
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=6)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    from isaaclab.app import AppLauncher
    class Launcher(AppLauncher):
        def _resolve_device_settings(self, values):
            super()._resolve_device_settings(values)
            values['active_gpu']=args.gpu;values['multi_gpu']=False
    app=Launcher(dict(headless=True,enable_cameras=True,device='cpu')).app
    try:
        torch.set_num_threads(2)
        with torch.inference_mode():capture(args)
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    app.close()
