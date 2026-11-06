"""Validate randomized sphere reaching and export two-camera review videos."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def preview(args):
    import numpy as np
    import torch
    from PIL import Image
    from vig_dagger.toy_reach_env.sphere_config import SphereReachEnvCfg
    from vig_dagger.toy_reach_env.sphere_environment import SphereReachEnv
    from vig_dagger.toy_evaluation import video
    cfg = SphereReachEnvCfg()
    env = SphereReachEnv(cfg)
    results, all_paths, endpoints = [], [], []
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [dict(seed=5100+i, kind='random') for i in range(args.episodes)]
    center = np.asarray(cfg.sphere_center)
    for axis in range(3):
        for sign in (-1, 1):
            delta = np.eye(3)[axis]*cfg.sphere_radius*sign
            cases.append(dict(seed=5200+len(cases), kind='diameter_control',
                              point_a=(center+delta).tolist(), point_b=(center-delta).tolist()))
    try:
        obs, _ = env.reset(seed=5100)
        initial = obs['policy'].clone()
        a, b = cfg.point_a, cfg.point_b
        obs, _ = env.reset(seed=5100)
        assert np.array_equal(a, cfg.point_a) and np.array_equal(b, cfg.point_b)
        reset_error = float((obs['policy']-initial).abs().max())
        assert reset_error < 1e-4, reset_error
        for index, case in enumerate(cases):
            options = {k:case[k] for k in ('point_a','point_b') if k in case}
            obs, _ = env.reset(seed=case['seed'], options=options)
            a, b = np.array(cfg.point_a), np.array(cfg.point_b)
            plan = env.sample_plan(case['seed']+100000)
            path, actions, frames = [], [], []
            stable = 0; max_line_distance = 0.
            for step in range(1400):
                state = obs['policy'][0].cpu().numpy().copy()
                assert state.shape == (12,)
                assert np.allclose(state[9:12], b, atol=1e-7)
                assert np.isfinite(state).all()
                path.append(state[:3])
                if step % 2 == 0:
                    images = [obs['rgb_camera'][f'{c}_cam'][0].cpu().numpy().copy()
                              for c in ('third_person','wrist')]
                    assert all(im.shape==(256,256,3) and im.dtype==np.uint8 and im.std()>2 for im in images)
                    frames.append((*images, f'Sphere reach {index+1} | {case["kind"]} | step {step}'))
                target = plan.target(state[:3], lookahead=.012)
                action = env.action_toward(target, max_translation=.008)
                actions.append(action[0].cpu().numpy().copy())
                obs, _ = env.step(action)
                error = float(obs['task']['goal_distance'][0,0])
                max_line_distance = max(max_line_distance, float(obs['task']['line_distance'][0,0]))
                stable = stable+1 if error < .0015 else 0
                if stable >= 15:
                    break
            result = dict(case, point_a=a.tolist(), point_b=b.tolist(), steps=step+1,
                          success=stable>=15, goal_error_m=error, max_line_distance_m=max_line_distance,
                          start_error_m=float(np.linalg.norm(path[0]-a)))
            for name, expected in [('start_marker',a),('goal_marker',b)]:
                actual, _ = env.scene[name].get_world_poses()
                assert np.allclose(actual[0].cpu().numpy(),expected,atol=1e-6)
            result['marker_positions_verified'] = True
            final_images = [obs['rgb_camera'][f'{c}_cam'][0].cpu().numpy().copy() for c in ('third_person','wrist')]
            frames.extend([(*final_images, f'Sphere reach {index+1} | error {error*1000:.2f} mm')]*15)
            video(args.out/f'episode_{index:02d}.mp4',frames)
            if index == 0:
                Image.fromarray(np.concatenate(final_images,axis=1)).save(args.out/'camera_preview.png')
            np.savez_compressed(args.out/f'episode_{index:02d}.npz',positions=np.array(path),actions=np.array(actions),
                                point_a=a,point_b=b,control_points=plan.control_points)
            results.append(result);all_paths.append(np.array(path));endpoints.extend([a,b])
            (args.out/'episodes.json').write_text(json.dumps(results,indent=2))
            print('SPHERE_EPISODE '+json.dumps(result),flush=True)
            assert result['success'], result
            assert max_line_distance < cfg.tube_radius+.004, result
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(9,8));ax=fig.add_subplot(111,projection='3d')
        u,v=np.meshgrid(np.linspace(0,2*np.pi,32),np.linspace(0,np.pi,16))
        ax.plot_wireframe(center[0]+cfg.sphere_radius*np.cos(u)*np.sin(v),
                          center[1]+cfg.sphere_radius*np.sin(u)*np.sin(v),
                          center[2]+cfg.sphere_radius*np.cos(v),color='gray',alpha=.2,linewidth=.4)
        for path,result in zip(all_paths,results):
            if result['kind']!='random':continue
            line,=ax.plot(*path.T,lw=1.5)
            ax.scatter(*result['point_a'],marker='o',color=line.get_color(),s=40)
            ax.scatter(*result['point_b'],marker='*',color=line.get_color(),s=90)
        ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Random sphere endpoints: circles = A, stars = B\nActual EEF trajectories with narrow variations around each chord')
        ax.set_box_aspect((1,1,1));fig.tight_layout();fig.savefig(args.out/'sphere_trajectories.png',dpi=160);plt.close(fig)
        summary=dict(sphere_center=cfg.sphere_center,sphere_radius_m=cfg.sphere_radius,
                     min_endpoint_separation_m=cfg.min_endpoint_separation,episodes=len(results),
                     successes=sum(r['success'] for r in results),seeded_reset_max_error=reset_error,
                     policy_state_order='TCP xyz, quaternion wxyz, two finger positions, goal xyz',
                     policy_state_shape=[1,12],camera_shape=[1,256,256,3],path='Bezier narrow tube around straight chord; endpoints fixed',
                     worst_goal_error_m=max(r['goal_error_m'] for r in results),
                     worst_line_distance_m=max(r['max_line_distance_m'] for r in results))
        (args.out/'validation.json').write_text(json.dumps(summary,indent=2))
        print('SPHERE_VALIDATED '+json.dumps(summary),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--episodes',type=int,default=8)
    args=parser.parse_args()
    if args.episodes < 1:parser.error('Need at least one random episode')
    if args.out.exists() and any(args.out.iterdir()):parser.error('Use a fresh output directory')
    from isaaclab.app import AppLauncher
    class Launcher(AppLauncher):
        def _resolve_device_settings(self,values):
            super()._resolve_device_settings(values)
            values['active_gpu']=args.gpu;values['multi_gpu']=False
    app=Launcher(dict(headless=True,enable_cameras=True,device='cpu')).app
    try:
        import torch
        torch.set_num_threads(2)
        with torch.inference_mode():preview(args)
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    app.close()
