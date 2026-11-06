"""Capture matched simulator snapshots, then evaluate selected states repeatedly."""
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


def snapshot(obs):
    return (obs['policy'][0].cpu().numpy().copy(),
            np.stack([obs['rgb_camera'][f'{c}_cam'][0].cpu().numpy().copy()
                      for c in ('third_person', 'wrist')]))


def write_tree(group, data):
    for key, value in data.items():
        if isinstance(value, dict):
            write_tree(group.create_group(key), value)
        else:
            group.create_dataset(key, data=value.cpu().numpy())


def read_tree(group, device):
    return {key: read_tree(value, device) if isinstance(value, h5py.Group)
            else torch.tensor(value[:], device=device) for key, value in group.items()}


def restore(env, scene):
    env.reset_to(scene, env_ids=None, seed=11000, is_relative=True)
    # Isaac's scene.reset_to copies physical joint velocities into desired PD
    # velocities. Our normal position-control task always has zero desired
    # velocity; leaving the copied targets adds a persistent unintended drive.
    # Preserve the saved physical velocities, but restore the controller target.
    robot = env.scene['robot']
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
    env.scene.write_data_to_sim()
    for _ in range(3):
        env.sim.render()
    for name in ('third_person_cam', 'wrist_cam'):
        env.scene[name].update(env.physics_dt, force_recompute=True)
    return env.observation_manager.compute(update_history=True)


def capture(args, env):
    records, states, images, references = [], [], [], []
    source = Path('/cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_validation_5_v1/dataset/toy_reach/shards')
    for path in sorted(source.glob('*.hdf5')):
        with h5py.File(path) as f:
            g = next(iter(f['data'].values()))
            for t in np.linspace(0, len(g['actions'])-1, 20).astype(int):
                states.append(g['obs/state'][t])
                images.append(np.stack([g[f'obs/rgb_camera/{c}_cam'][t] for c in ('third_person', 'wrist')]))
                references.append(g['actions'][t, :3])
                records.append(dict(kind='calibration', source=path.name, frame=int(t)))
    a, b = np.array(env.cfg.point_a), np.array(env.cfg.point_b)
    cases = [(0., 0., 0)]
    for progress in (.5, .9, .98, 1.02):
        cases.append((progress, 0., 0))
        for radius in (.0025, .005, .01, .02):
            cases.extend((progress, radius, direction) for direction in range(4))
    with h5py.File(args.root/'observations.tmp.hdf5', 'w') as f:
        scenes = f.create_group('scenes')
        for probe, (progress, radius, direction) in enumerate(cases):
            angle = direction*np.pi/2
            position = a + progress*(b-a) + radius*np.array([np.cos(angle), 0, np.sin(angle)])
            env.reset(seed=11000, options={'eef_position':position})
            scene = env.scene.get_state(is_relative=True)
            obs = restore(env, scene)
            state, rgb = snapshot(obs)
            write_tree(scenes.create_group(str(len(records))), scene)
            references.append(env.action_toward(b, max_translation=.008)[0, :3].cpu().numpy().copy())
            records.append(dict(kind='probe', probe=probe, progress=progress, radius=radius,
                                direction=direction, requested_position=position.tolist(),
                                placement_error=float(np.linalg.norm(state[:3]-position))))
            states.append(state); images.append(rgb)
            print(f'CAPTURE {probe+1}/{len(cases)} progress={progress} radius={radius}', flush=True)
        f.create_dataset('states', data=np.array(states))
        f.create_dataset('images', data=np.array(images), compression='gzip', compression_opts=1)
        f.create_dataset('references', data=np.array(references))
        f.attrs['records'] = json.dumps(records)
    (args.root/'observations.tmp.hdf5').rename(args.root/'observations.hdf5')
    (args.root/'capture_done.json').write_text(json.dumps({'calibration':100, 'probes':len(cases)}))


def validate_restore(args, env):
    """Replay identical controls before/after restoring a physical snapshot."""
    results = []
    for progress in (0., .5, .98):
        a, b = np.array(env.cfg.point_a), np.array(env.cfg.point_b)
        env.reset(seed=11000, options={'eef_position':a+progress*(b-a)})
        scene = env.scene.get_state(is_relative=True)
        direct, actions = [], []
        for _ in range(80):
            action = env.action_toward(b, max_translation=.008)
            actions.append(action.clone())
            obs, _ = env.step(action)
            direct.append(obs['policy'][0].clone())
        restore(env, scene)
        replay = []
        for action in actions:
            obs, _ = env.step(action)
            replay.append(obs['policy'][0].clone())
        difference = (torch.stack(direct)-torch.stack(replay)).abs()
        result = dict(progress=progress, max_xyz_difference_m=float(difference[:, :3].max()),
                      max_state_difference=float(difference.max()))
        results.append(result)
        print('RESTORE_CHECK '+json.dumps(result), flush=True)
        if result['max_xyz_difference_m'] > 1e-4 or result['max_state_difference'] > 1e-3:
            raise RuntimeError(f'Restore replay differs from direct placement: {result}')
    (args.root/'restore_validation.json').write_text(json.dumps(results, indent=2))


def reference_controls(args, env):
    indices = sorted({row['index'] for path in args.root.glob('demos*/selection.json')
                      for row in json.loads(path.read_text())})
    results = []
    env.reset(seed=11000, options={'randomize_start':False})
    with h5py.File(args.root/'observations.hdf5') as f:
        for index in indices:
            obs = restore(env, read_tree(f[f'scenes/{index}'], env.device))
            initial = obs['policy'][0, :3].cpu().numpy().copy()
            stable = 0
            for step in range(1200):
                obs, _ = env.step(env.action_toward(env.cfg.point_b, max_translation=.008))
                position = obs['policy'][0, :3].cpu().numpy()
                error = float(np.linalg.norm(position-np.array(env.cfg.point_b)))
                stable = stable+1 if error < .005 else 0
                if stable >= 15:
                    break
            result = dict(index=index, success=stable>=15, steps=step+1,
                          initial_position=initial.tolist(), final_position=position.tolist(), goal_error_m=error)
            results.append(result)
            print('REFERENCE '+json.dumps(result), flush=True)
            (args.root/'reference_controls.json').write_text(json.dumps(results, indent=2))
    (args.root/'reference_controls_done.json').write_text(json.dumps({'states':len(results),'successes':sum(r['success'] for r in results)}))


def rollouts(args, env):
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_evaluation import observation, video
    from vig_dagger.toy_flow_search import sample_and_score
    out = args.root/args.run_dir.name
    policy = LibraryPolicy(args.run_dir/'latest.pt', 'cuda:0')
    selections = json.loads((out/'selection.json').read_text())
    results = []
    path = out/'rollouts.json'
    if path.exists():
        results = json.loads(path.read_text())
    done = {(x['index'], x['trial']) for x in results}
    env.reset(seed=11000, options={'randomize_start':False})
    start = time.time()
    with h5py.File(args.root/'observations.hdf5') as f:
        for selection in selections:
            index = selection['index']
            scene = read_tree(f[f'scenes/{index}'], env.device)
            expected = f['states'][index]
            for trial in range(8):
                if (index, trial) in done:
                    continue
                obs = restore(env, scene)
                initial_state, initial_rgb = snapshot(obs)
                state_error = float(np.max(np.abs(initial_state-expected)))
                if state_error > 1e-4:
                    raise RuntimeError(f'Snapshot restore differs at {index}: {state_error}')
                seed = 21000 + index*100 + trial
                s, im = observation(obs, policy.device)
                # The scored chunk below is the chunk actually executed first.
                _, raw, flow, flow_se = sample_and_score(policy, s, im, seed, samples=1, draws=512)
                chunk = torch.from_numpy(raw[0]).to(policy.device)
                torch.manual_seed(seed+1000000)
                states, actions, frames = [], [], []
                reason = 'time_limit'; stable = 0; min_error = float('inf')
                for step in range(1200):
                    state = obs['policy'][0].cpu().numpy().copy()
                    states.append(state)
                    if trial == 0 and step % 2 == 0:
                        _, rgb = snapshot(obs)
                        frames.append((rgb[0], rgb[1], f'{args.run_dir.name} | state {index} | step {step}'))
                    if step and step % 8 == 0:
                        s, im = observation(obs, policy.device)
                        chunk = policy.act(s, im, factor=1)[0]
                    action = chunk[step % 8].reshape(1, 7).to(env.device)
                    if not torch.isfinite(action).all():
                        reason = 'nonfinite_action'; break
                    if action[0, :3].norm() > .05 or action[0, 3:6].norm() > .2:
                        reason = 'extreme_action'; break
                    actions.append(action[0].cpu().numpy().copy())
                    obs, _ = env.step(action)
                    p = obs['policy'][0, :3].cpu().numpy()
                    error = float(np.linalg.norm(p-np.array(env.cfg.point_b)))
                    min_error = min(min_error, error)
                    stable = stable+1 if error < .005 else 0
                    if stable >= 15:
                        reason = 'success'; break
                    if not (.15 < p[0] < .75 and -.4 < p[1] < .45 and .12 < p[2] < .65):
                        reason = 'workspace_exit'; break
                final, rgb = snapshot(obs)
                error = float(np.linalg.norm(final[:3]-np.array(env.cfg.point_b)))
                key = f'state_{index:03d}_trial_{trial:02d}'
                if frames:
                    frames.extend([(rgb[0], rgb[1], f'{args.run_dir.name} | {reason} | error {error*1000:.1f} mm')]*15)
                    video(out/'videos'/f'{key}.mp4', frames)
                np.savez_compressed(out/f'{key}.npz', states=np.array(states), actions=np.array(actions),
                                    final_state=final, initial_images=initial_rgb, first_chunk=raw[0])
                result = dict(index=index, trial=trial, seed=seed, success=reason=='success', reason=reason,
                              steps=len(actions), goal_error_m=error, min_goal_error_m=min_error,
                              first_chunk_flow=float(flow[0]), first_chunk_flow_se=float(flow_se[0]),
                              restore_max_state_error=state_error,
                              initial_rgb_mae=float(np.abs(initial_rgb.astype(float)-f['images'][index]).mean()))
                results.append(result)
                temporary = out/'rollouts.tmp.json'
                temporary.write_text(json.dumps(results, indent=2)); temporary.replace(path)
                (out/'rollout_progress.json').write_text(json.dumps(dict(done=len(results), total=8*len(selections),
                    elapsed_seconds=time.time()-start)))
                print('ROLLOUT '+json.dumps(result), flush=True)
    (out/'rollouts_done.json').write_text(json.dumps({'episodes':len(results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'rollouts', 'validate_restore', 'reference_controls'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    from isaaclab.app import AppLauncher
    class Launcher(AppLauncher):
        def _resolve_device_settings(self, values):
            super()._resolve_device_settings(values)
            values['active_gpu'] = args.gpu; values['multi_gpu'] = False
    app = Launcher(dict(headless=True, enable_cameras=True, device='cpu')).app
    try:
        from vig_dagger.toy_reach_env.config import ToyReachEnvCfg
        from vig_dagger.toy_reach_env.environment import ToyReachEnv
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            env = ToyReachEnv(ToyReachEnvCfg())
            try:
                {'capture':capture, 'rollouts':rollouts, 'validate_restore':validate_restore,
                 'reference_controls':reference_controls}[args.mode](args, env)
            finally:
                env.close()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    app.close()
