"""Collect insertion-only previews, retaining raw states and policy camera videos."""
import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import sapien
from PIL import Image, ImageDraw

from vig_dagger.dataset import convert, _ffmpeg_writer
from vig_dagger.envs import POLICY_CAMERAS
from vig_dagger.insertion_env import make_insertion_env, solve_insertion
from vig_dagger.recording import TaskStateLog, SplitRecordEpisode, EXPERT


class HeldStartStateLog(TaskStateLog):
    def _snapshot(self, first):
        # Reset has already advanced physics and verified the contact grasp.
        super()._snapshot(first=False)


def finish_video(proc):
    proc.stdin.close()
    error = proc.stderr.read().decode(errors='replace')
    if proc.wait():
        raise RuntimeError(error)


def export_previews(root):
    raw = root / 'raw' / 'preview.h5'
    meta = json.loads(raw.with_suffix('.json').read_text())
    metadata = {f"traj_{x['episode_id']}": x for x in meta['episodes']}
    video_dir = root / 'videos'
    video_dir.mkdir(exist_ok=True)
    combined = _ffmpeg_writer(video_dir / 'all_5_demos.mp4', 328, 768)
    sheets = []
    validation = []
    try:
        with h5py.File(raw) as h5:
            keys = sorted(h5, key=lambda k: int(k.split('_')[-1]))
            for index, key in enumerate(keys):
                group = h5[key]
                steps = len(group['actions'])
                state = group['task_state']
                assert group['actions'].shape == (steps, 8)
                assert group['obs/agent/qpos'].shape == (steps + 1, 9)
                assert state['held'][0] and state['held'][-1]
                assert np.all(group['success'][-10:])
                assert np.isfinite(group['actions'][:]).all()
                initial = metadata[key]['initialization']
                stages = metadata[key]['stages']
                proc = _ffmpeg_writer(video_dir / f'demo_{index:02d}.mp4', 328, 768)
                start_end = []
                for t in range(steps + 1):
                    frame = Image.new('RGB', (768, 328), (23, 27, 34))
                    draw = ImageDraw.Draw(frame)
                    for ci, camera in enumerate(POLICY_CAMERAS):
                        rgb = group[f'obs/sensor_data/{camera}/rgb']
                        assert len(rgb) == steps + 1
                        frame.paste(Image.fromarray(rgb[t]), (256 * ci, 24))
                        draw.text((256 * ci + 8, 6), camera, fill='white')
                    stage = next((s['stage'] for s in stages if s['start_step'] <= t < s['end_step']), 'hold')
                    draw.text((8, 286), f'Demo {index+1} | frame {t}/{steps} | {stage} | distance {1000*state["dist"][t]:.1f} mm | angle {np.rad2deg(state["angle"][t]):.1f} deg', fill='white')
                    rpy = initial['requested_rpy_deg']
                    draw.text((8, 306), f'Start: {100*initial["requested_standoff_m"]:.1f} cm from seated pose; roll/pitch/yaw {rpy[0]:+.1f}/{rpy[1]:+.1f}/{rpy[2]:+.1f} deg', fill='white')
                    payload = np.asarray(frame).tobytes()
                    proc.stdin.write(payload)
                    combined.stdin.write(payload)
                    if t in (0, steps):
                        start_end.append(frame.copy())
                finish_video(proc)
                sheets.extend(start_end)
                validation.append(dict(demo=index, steps=steps, initial_held=bool(state['held'][0]),
                    final_held=bool(state['held'][-1]), held_all_steps=bool(np.all(state['held'][:])),
                    final_distance_mm=float(state['dist'][-1]*1000),
                    final_angle_deg=float(np.rad2deg(state['angle'][-1])),
                    last_10_steps_success=bool(np.all(group['success'][-10:]))))
    finally:
        finish_video(combined)
    sheet = Image.new('RGB', (1536, 328 * (len(sheets)//2)))
    for i, frame in enumerate(sheets):
        sheet.paste(frame, ((i%2)*768, (i//2)*328))
    sheet.save(root / 'start_and_end.png')
    convert([(raw, None)], root / 'dataset')
    # Decode every deliverable, including the separate training camera streams.
    for path in sorted(root.rglob('*.mp4')):
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    (root / 'validation.json').write_text(json.dumps(validation, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--num-demos', type=int, default=5)
    parser.add_argument('--start-seed', type=int, default=100)
    parser.add_argument('--max-attempts', type=int, default=20)
    parser.add_argument('--render-backend', default='cuda:0')
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--bulk', action='store_true', help='Export raw data and camera streams without review montages')
    parser.add_argument('--trajectory-name', default='preview')
    args = parser.parse_args()
    device = sapien.Device(args.render_backend)
    expected = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=pci.bus_id', '--format=csv,noheader'], text=True).strip().lower()
    expected = expected[-12:]
    if device.pci_string.lower() != expected:
        raise RuntimeError(f'Expected physical GPU {args.gpu} ({expected}), got {device.pci_string}')
    print(f'Rendering on physical GPU {args.gpu}: {device.pci_string}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'raw' / f'{args.trajectory_name}.h5').exists():
        raise FileExistsError('Refusing to overwrite an existing collection')
    start = time.time()
    log = HeldStartStateLog(make_insertion_env(render_backend=args.render_backend))
    recorder = SplitRecordEpisode(log, success_dir=args.out/'raw', fail_dir=args.out/'failed',
        trajectory_name=args.trajectory_name, source_type='motionplanning',
        source_desc='Insertion-only preview: physical grasp initialized near socket; main-branch scene unchanged',
        record_reward=False)
    rows = []
    success_count = 0
    try:
        for attempt in range(args.max_attempts):
            seed = args.start_seed + attempt
            print(f'BEGIN seed={seed}', flush=True)
            reset_ok = False
            row = dict(seed=seed, success=False)
            try:
                _, info = recorder.reset(seed=seed)
                reset_ok = True
                log.controller = EXPERT
                row['initialization'] = info['insertion_initialization']
                result, boundaries = solve_insertion(recorder, log)
                states = log.arrays()
                success = (bool(result[-1]['success'][0]) and not bool(result[3][0])
                    and np.all(states['held']) and np.all(states['dist'][-10:] <= .005)
                    and np.all(states['angle'][-10:] <= .2))
                row.update(success=bool(success), steps=log.num_steps, stages=boundaries)
                recorder.flush_episode(success=success, save=True, extra_info=row,
                    extra_groups={'task_state': states})
                success_count += int(success)
            except Exception as exc:
                row['error'] = repr(exc)
                traceback.print_exc()
                if reset_ok and log.num_steps:
                    recorder.flush_episode(success=False, save=True, extra_info=row,
                        extra_groups={'task_state': log.arrays()})
            rows.append(row)
            (args.out/'attempts.json').write_text(json.dumps(rows, indent=2))
            print(f'END seed={seed}, success={row["success"]}, collected={success_count}/{args.num_demos}', flush=True)
            if success_count == args.num_demos:
                break
    finally:
        recorder.close()
    if success_count != args.num_demos:
        raise RuntimeError(f'Only {success_count} of {args.num_demos} successful demos')
    if args.bulk:
        raw = args.out/'raw'/f'{args.trajectory_name}.h5'
        with h5py.File(raw) as h5:
            assert len(h5) == args.num_demos
            for g in h5.values():
                n = len(g['actions'])
                assert g['actions'].shape == (n, 8) and np.isfinite(g['actions'][:]).all()
                assert g['obs/agent/qpos'].shape == (n+1, 9)
                assert np.all(g['task_state/held'][:]) and np.all(g['success'][-10:])
                for camera in POLICY_CAMERAS:
                    assert g[f'obs/sensor_data/{camera}/rgb'].shape == (n+1,256,256,3)
        convert([(raw, None)], args.out/'dataset')
        for path in sorted((args.out/'dataset').rglob('*.mp4')):
            subprocess.run(['ffmpeg','-threads','2','-v','error','-xerror','-i',str(path),'-f','null','-'],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    else:
        export_previews(args.out)
    summary = dict(demos=success_count, attempts=len(rows), seconds=time.time()-start,
        main_commit='5d2ab39603bbb1269b5129ce6e9705df52ae7add',
        gpu=args.gpu, cameras=list(POLICY_CAMERAS), camera_resolution=256,
        control_hz=20, action_mode='pd_joint_pos',
        initial_standoff_m=[.03,.08], lateral_yz_bounds_m=[-.015,.015],
        roll_pitch_yaw_bounds_deg=[-8,8], success_distance_m=.005, success_angle_rad=.2)
    (args.out/'complete.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
