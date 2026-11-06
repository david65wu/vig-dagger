"""Detached queue: calibrate, replay stored actions, and render four-score videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time

from run_insertion_experiment import REPO,run_job,write_json

BASE=Path('/cpfs02/user/caojianuo/vig-dagger-runs')
SOURCE=BASE/'charger_insertion_moe7_three_scores_v1'
EPOCHS=[10,25,50,100,200,300]


def publish(root,calibration_only=False):
    # Importing the renderer also imports torch; done once by the coordinator.
    from render_insertion_lowest_reference import publish as gallery
    gallery(root)
    done=sum((p/'complete.json').exists() for p in (root/'runs').glob('demos*_epoch*'))
    scored=sum((p/'scoring_complete.json').exists() for p in (root/'runs').glob('demos*_epoch*'))
    calibrated=sum((p/'calibration_complete.json').exists() for p in (root/'runs').glob('demos*_epoch*'))
    write_json(root/'status.json',dict(phase='calibrating_only' if calibration_only else 'processing',calibrated_checkpoints=calibrated,completed_checkpoints=done,scored_checkpoints=scored,target_checkpoints=18,target_episodes=900,target_mp4s=1800,updated_at=time.time()))


def coordinate(root,calibration_only=False):
    if not calibration_only:assert json.loads((root/'preflight.json').read_text())['passed']
    names=[f'demos{n}_epoch{e:03d}' for n in [500,200,50] for e in EPOCHS]
    root.mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    for name in names:(root/'runs'/name).mkdir(parents=True,exist_ok=True)
    write_json(root/'protocol.json',dict(source=str(SOURCE),method='Replay saved initial states and executed actions; no new policy decisions',
        validation='Every frame must reproduce original and chunk-mean VIG within atol=1e-6, rtol=1e-4; TCP translation and plug distance within 1e-5 m; TCP quaternion within 1e-5; plug angle within 1e-3 rad; success label must agree',
        reference_factor=64,comparison_factors=[1,2,4,8,16,32],samples_per_level=32,aggregation='Maximum of mean per-cell 1D Wasserstein; original 16 x 8 action cells',
        seed_protocol='Same VIG seeds as source; all variants computed from identical regenerated samples',
        calibration='All pre-action training-demo frames of this checkpoint own 50/200/500 subset; independent empirical percentile',
        video_ema_alpha=.1,video_size=[512,1020],video_panels=['flow','full-reference VIG','chunk-mean VIG','lowest-reference VIG'],
        raw_axis_scope='Per episode, per metric; fixed over video',summary_smoothing=False,gpus=list(range(8)),
        calibration_only=calibration_only,replay_method_pending=calibration_only))
    provenance=root/'provenance';provenance.mkdir(exist_ok=True)
    for f in ['scripts/run_insertion_lowest_reference.py','scripts/rescore_insertion_lowest_reference.py','scripts/render_insertion_lowest_reference.py','scripts/render_insertion_raw_videos.py','src/vig_dagger/scoring.py','PLUG_INSERTION_PLAN.md']:
        shutil.copy2(REPO/f,provenance/Path(f).name)
    pending=queue.Queue()
    # Render a fully calibrated smaller checkpoint early for inspection.
    names.remove('demos50_epoch300');names.insert(0,'demos50_epoch300')
    for name in names:pending.put(name)
    publish(root,calibration_only)
    def lane(gpu):
        while True:
            try:name=pending.get_nowait()
            except queue.Empty:return
            out=root/'runs'/name
            if calibration_only:
                if not (out/'calibration_complete.json').exists():
                    run_job([sys.executable,'-u','scripts/rescore_insertion_lowest_reference.py','--root',str(root),'--name',name,'--calibration-only'],root/'logs'/f'{name}_calibration.log',gpu,root/'processes'/f'{name}_calibration.json',f'--root {root} --name {name} --calibration-only')
                assert (out/'calibration_complete.json').exists()
                continue
            if not (out/'scoring_complete.json').exists():
                run_job([sys.executable,'-u','scripts/rescore_insertion_lowest_reference.py','--root',str(root),'--name',name],root/'logs'/f'{name}.log',gpu,root/'processes'/f'{name}.json',f'--root {root} --name {name}')
            assert (out/'scoring_complete.json').exists()
            if not (out/'complete.json').exists():
                with (root/'logs'/f'{name}_render.log').open('a') as log:
                    subprocess.run([sys.executable,'-u','scripts/render_insertion_lowest_reference.py','--root',str(root),'--name',name],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert (out/'complete.json').exists()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,g) for g in range(8)]
        # Publish incremental progress while jobs run; failures remain visible
        # in per-checkpoint logs and are raised by the final future checks.
        while any(not f.done() for f in futures):
            publish(root,calibration_only)
            for f in futures:
                if f.done():f.result()
            time.sleep(15)
        for f in futures:f.result()
    publish(root,calibration_only)
    if calibration_only:
        write_json(root/'status.json',dict(phase='calibration_complete',checkpoints=18,replay_method_pending=True,finished_at=time.time()))
        return
    subprocess.run([sys.executable,'scripts/render_insertion_lowest_reference.py','--root',str(root),'--plot'],cwd=REPO,check=True)
    write_json(root/'status.json',dict(phase='complete',checkpoints=18,episodes=900,mp4s=1800,new_policy_rollouts=0,finished_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--calibration-only',action='store_true');a=p.parse_args()
    with (a.root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:coordinate(a.root,a.calibration_only)
        except BaseException as exc:
            write_json(a.root/'status.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
