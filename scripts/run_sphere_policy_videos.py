"""Fresh sphere rollouts, training-flow calibration, and full-chunk annotated videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from run_sphere_rollouts import NAMES, write, report

REPO = Path(__file__).resolve().parents[1]


def annotate(args):
    import cv2
    import torch
    from annotate_sphere_rollouts import decoded_inputs, render_video, cumulative_command_paths
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_flow_search import sample_and_score

    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out = args.root / args.name
    dest = out / 'calibrated_chunk_videos'; dest.mkdir(exist_ok=True)
    checkpoint = args.training / 'policies' / args.name / args.checkpoint
    policy = LibraryPolicy(checkpoint, f'cuda:{args.gpu}')
    camera = json.loads((args.root / 'camera_calibration.json').read_text())
    calibration = json.loads((args.root / 'flow_calibration' / f'{args.name}.json').read_text())
    assert calibration['checkpoint'] == str(checkpoint)
    rows = json.loads((out / 'rollouts.json').read_text())
    started = time.time()
    for count, row in enumerate(rows):
        stem = f"{row['case_id']:02d}_{row['kind']}"
        target = dest / f'{stem}.mp4'
        if target.exists() and target.with_suffix('.json').exists():continue
        with np.load(out / f'{stem}.npz') as trace:
            states = trace['states']; actions = trace['actions']
        # Even an immediate action-limit abort has an initial observation to inspect.
        steps = list(range(0, max(1, len(actions)), 8))
        images = decoded_inputs(out / row['video'], steps)
        chunks = []; flows = []; losses = []
        for index, step in enumerate(steps):
            state = torch.tensor(states[step], device=policy.device)[None]
            rgb = torch.tensor(images[step], device=policy.device).permute(0, 3, 1, 2)[None]
            values, raw, loss, _ = sample_and_score(policy, state, rgb,
                seed=91000 + row['case_id'] * 10000 + step, samples=32, draws=64)
            chunks.append(raw); flows.append(values['flow_mean']); losses.append(loss)
            if index % 25 == 0:
                write(out / 'annotation_progress.json', dict(completed=count, total=len(rows),
                    case_id=row['case_id'], scored_frames=index+1, total_frames=len(steps),
                    elapsed_seconds=time.time()-started))
        chunks = np.asarray(chunks); flows = np.asarray(flows); losses = np.asarray(losses)
        assert chunks.shape == (len(steps), 32, 16, 7)
        assert np.isfinite(chunks).all() and np.isfinite(flows).all()
        assert np.allclose(flows, losses.mean(axis=1))
        paths = cumulative_command_paths(states[steps, None, None, :3], chunks[:, :, :, :3])
        assert np.allclose(np.diff(paths, axis=2), chunks[:, :, :, :3], atol=1e-7)
        np.savez_compressed(dest / f'{stem}_chunks.npz', steps=steps, flow=flows,
                            sampled_chunks=chunks, chunk_losses=losses)
        temporary = target.with_name(stem + '.partial.mp4')
        # A failed encoder may leave a partial file; preserve it before retrying.
        if temporary.exists():temporary.rename(temporary.with_name(stem+f'.incomplete-{time.time_ns()}.mp4'))
        render_video(temporary, images, steps, states, actions, chunks[:, :, :, :3], flows,
                     camera, args.name + ' | ' + row['kind'] + f" | case {row['case_id']}",
                     row['reason'], full_chunks=True, training_p95=calibration['p95'])
        temporary.replace(target); temporary.with_suffix('.png').replace(target.with_suffix('.png'))
        write(target.with_suffix('.json'), dict(case_id=row['case_id'], checkpoint=str(checkpoint),
            samples=32, horizon=16, noise_time_draws_per_chunk=64, every_control_steps=8, magnification=1,
            orange='All sixteen cumulative base-frame translations from each of 32 newly sampled chunks',
            cyan='Up to eight commands actually executed before the next replan',
            path='Nominal command paths, not simulated future EEF trajectories',
            flow_source='Exact recorded state plus decoded original H264 camera frames; recomputed with this checkpoint',
            training_flow_median=calibration['median'], training_flow_p95=calibration['p95'],
            training_calibration_frames=calibration['frames'], outcome=row['reason'], success=row['success']))
        write(out / 'annotation_progress.json', dict(completed=count+1, total=len(rows),
              elapsed_seconds=time.time()-started))
        print('ANNOTATED ' + stem, flush=True)
    write(out / 'annotation_done.json', dict(episodes=len(rows), checkpoint=str(checkpoint)))


def validate_video(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate',
        '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    assert not result.stderr.strip(), (path, result.stderr)
    stream = json.loads(result.stdout)['streams'][0]
    meta = json.loads(path.with_suffix('.json').read_text())
    with np.load(path.with_name(path.stem+'_chunks.npz')) as data:
        assert data['sampled_chunks'].shape == (len(data['steps']), 32, 16, 7)
        assert np.isfinite(data['sampled_chunks']).all() and np.isfinite(data['flow']).all()
        assert np.allclose(data['flow'], data['chunk_losses'].mean(axis=1))
        assert np.array_equal(data['steps'], np.arange(0, 8*len(data['steps']), 8))
        assert int(stream['nb_read_frames']) == len(data['steps']) + 8
    assert stream['width']==1024 and stream['height']==800 and stream['r_frame_rate']=='15/2'
    calibration = json.loads((path.parents[2]/'flow_calibration'/f'{path.parents[1].name}.json').read_text())
    assert meta['checkpoint'] == calibration['checkpoint'] and meta['horizon'] == 16
    assert meta['training_flow_p95'] == calibration['p95']
    return dict(path=str(path), decoded_frames=int(stream['nb_read_frames']), horizon=16,
                checkpoint=meta['checkpoint'], training_p95=meta['training_flow_p95'])


def prepare(args):
    from calibrate_sphere_flow import prepare as prepare_calibration
    root = args.root; source = args.previous
    (root/'logs').mkdir(exist_ok=True)
    for filename in ('cases.json', 'camera_calibration.json'):
        shutil.copy2(source/filename, root/filename)
    cases = json.loads((root/'cases.json').read_text())
    assert len(cases)==30 and {c['case_id'] for c in cases}==set(range(30))
    protocol = json.loads((source/'protocol.json').read_text())
    protocol.pop('reused_rollouts', None)
    protocol.update(checkpoint=args.checkpoint, training_root=str(args.training),
        case_source=str(source/'cases.json'), fresh_rollouts=True,
        camera_calibration_source=str(source/'camera_calibration.json'),
        camera_note='Same environment/cameras; reuse previously renderer-verified geometry only',
        video_arrows='All 16 cumulative translation commands of 32 sampled chunks, scale 1x; cyan up to 8 executed commands',
        flow_reference='Recomputed on each new policy own training selection: 20 frames/demo, 32 chunks x 64 draws',
        checkpoint_sha256={name: hashlib.sha256((args.training/'policies'/name/args.checkpoint).read_bytes()).hexdigest()
                           for name in NAMES})
    write(root/'protocol.json', protocol)
    prepare_calibration(root)
    calibration_protocol = json.loads((root/'flow_calibration/protocol.json').read_text())
    calibration_protocol.update(training_root=str(args.training), checkpoint=args.checkpoint)
    write(root/'flow_calibration/protocol.json', calibration_protocol)
    for relative in ['scripts/run_sphere_policy_videos.py', 'scripts/rollout_sphere_simple.py',
                     'scripts/calibrate_sphere_flow.py', 'scripts/annotate_sphere_rollouts.py',
                     'scripts/add_sphere_flow_reference.py', 'scripts/run_sphere_rollouts.py',
                     'scripts/toy_flow_search_sim.py', 'src/vig_dagger/toy_projection.py',
                     'src/vig_dagger/toy_flow_search.py', 'src/vig_dagger/policy.py',
                     'src/vig_dagger/toy_reach_env/environment.py',
                     'src/vig_dagger/toy_reach_env/sphere_environment.py']:
        dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,dest)
    write(root/'prepared.json', dict(training=str(args.training), checkpoint=args.checkpoint))


def run(args):
    from calibrate_sphere_flow import report as calibration_report
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root/'prepared.json').exists():prepare(args)
        assert json.loads((root/'prepared.json').read_text())==dict(training=str(args.training),checkpoint=args.checkpoint)
        setup=root.parent/'setup'
        env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',PYTHONUNBUFFERED='1',
            PYTHONHASHSEED='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
            VK_ICD_FILENAMES=str(setup/'nvidia-570/nvidia_icd_local.json'),
            __EGL_VENDOR_LIBRARY_FILENAMES=str(setup/'nvidia-570/10_nvidia_local.json'))
        env['LD_LIBRARY_PATH']=str(setup/'nvidia-graphics')+':'+env.get('LD_LIBRARY_PATH','')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        stages={}; processes={}; started=time.time()
        write(root/'flow_calibration/status.json',dict(state='scoring',checkpoint=args.checkpoint))
        for gpu,name in enumerate(NAMES):
            stages[name]='calibrate'
        def launch(name,stage,gpu):
            common=['--root',str(root),'--name',name,'--gpu',str(gpu),
                    '--training',str(args.training),'--checkpoint',args.checkpoint]
            if stage=='calibrate':cmd=[sys.executable,str(REPO/'scripts/calibrate_sphere_flow.py'),'score',*common]
            elif stage=='rollout':cmd=[str(REPO/'.venv-isaac/bin/python'),str(REPO/'scripts/rollout_sphere_simple.py'),*common]
            else:cmd=[sys.executable,__file__,'annotate',*common]
            with (root/'logs'/f'{name}_{stage}.log').open('a') as log:
                p=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=p;print(f'{stage} {name} gpu={gpu} pid={p.pid}',flush=True)
        for gpu,name in enumerate(NAMES):launch(name,stages[name],gpu)
        calibration_finished=False
        while True:
            for gpu,name in enumerate(NAMES):
                if stages[name]=='complete':continue
                p=processes[name]
                if p.poll() is None:continue
                if p.returncode:raise RuntimeError(f'{name} stage {stages[name]} exited {p.returncode}; see logs')
                if stages[name]=='annotate':stages[name]='complete';continue
                stages[name]={'calibrate':'rollout','rollout':'annotate'}[stages[name]]
                launch(name,stages[name],gpu)
            if not calibration_finished and all(stage!='calibrate' for stage in stages.values()):
                calibration_report(root)
                write(root/'flow_calibration/status.json',dict(state='complete',policies=6,observations=1500,checkpoint=args.checkpoint))
                calibration_finished=True
            write(root/'pipeline.json',dict(state='running',pid=os.getpid(),elapsed_seconds=time.time()-started,
                stages=stages,pids={name:p.pid for name,p in processes.items()},
                annotated_videos=len(list(root.glob('demos*/calibrated_chunk_videos/*.json'))),total_videos=180))
            if all(stage=='complete' for stage in stages.values()):break
            time.sleep(10)
        report(root)
        write(root/'pipeline.json',dict(state='validating',pid=os.getpid(),annotated_videos=180,total_videos=180))
        videos=sorted(root.glob('demos*/calibrated_chunk_videos/*.mp4'));assert len(videos)==180
        with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(validate_video,videos))
        for name in NAMES:
            calibration=json.loads((root/'flow_calibration'/f'{name}.json').read_text())
            with np.load(root/'flow_calibration'/f'{name}.npz') as data:
                assert np.allclose(data['flow'],data['chunk_losses'].mean(axis=1))
                assert np.allclose(np.percentile(data['flow'],[50,95]),[calibration['median'],calibration['p95']])
        write(root/'calibrated_chunk_validation.json',dict(videos=180,all_frames_decoded=True,horizon=16,
            per_policy_training_reference=True,checkpoint=args.checkpoint,checks=checks))
        write(root/'pipeline.json',dict(state='complete',rollouts=180,annotated_videos=180,
            checkpoint=args.checkpoint,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','annotate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--checkpoint',default='epoch-300.pt');p.add_argument('--previous',type=Path)
    p.add_argument('--name');p.add_argument('--gpu',type=int)
    args=p.parse_args()
    try:globals()[args.mode](args)
    except BaseException as exc:
        if args.mode=='run':write(args.root/'pipeline.json',dict(state='failed',error=repr(exc),pid=os.getpid()))
        traceback.print_exc();raise
