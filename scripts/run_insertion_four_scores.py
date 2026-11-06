"""Fresh policy rollouts, four every-frame scores, and retained observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time

from run_insertion_experiment import REPO,write_json,run_job,alive
BASE=Path('/cpfs02/user/caojianuo/vig-dagger-runs')
SOURCE=BASE/'charger_insertion_moe7_three_scores_v1'
CALROOT=BASE/'charger_insertion_moe7_lowest_reference_v1'
NAMES=[f'demos{n}_epoch{e:03d}' for n in [50,200,500] for e in [10,25,50,100,200,300]]


def prepare(root):
    root.mkdir(exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'cases.json',root/'cases.json')
    for case in json.loads((root/'cases.json').read_text()):
        target=root/case['state_file'];target.parent.mkdir(exist_ok=True)
        if not target.exists():os.link(SOURCE/case['state_file'],target)
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    protocol.update(policy_label='7-level MoE | fresh rollouts | flow + three VIG variants',fresh_policy_rollouts=True,
        source_starting_cases=str(SOURCE),lowest_calibration_source=str(CALROOT),vig_lowest_reference_factor=64,
        vig_sampling='All three VIG variants use the identical 7 x 32 x 16 x 8 normalized action samples',
        full_rollout_videos=True,score_stride=1,calibration_stride=1,vig_enabled=True,vig_both=True,vig_temporal_average=True,
        video_panels=['flow','full-reference VIG','chunk-mean VIG','lowest-reference VIG'],video_size=[512,1020],
        video_ema_alpha=.1,raw_axis_scope='per episode, per metric; fixed through playback',
        retained_per_frame=['pre-action qpos','lossless hand/side RGB','camera calibration','all seven-resolution VIG samples','flow sampled chunk','executed action','sampling seeds'],
        calibration='Reuse hash-verified every-frame training references for existing scores; new lowest-reference VIG calibration on the same frames',
        aggregation='Lowest-reference VIG: maximum over factors 1,2,4,8,16,32 of mean per-cell 1D W1 versus factor 64, without temporal averaging')
    write_json(root/'protocol.json',protocol)


def prepare_run(root,name):
    import numpy as np
    out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    src=SOURCE/'runs'/name;new=CALROOT/'runs'/name
    assert (new/'calibration_complete.json').exists()
    for stem,parent in [('calibration',src),('calibration_vig',src),('calibration_vig_original',src),('calibration_vig_lowest',new)]:
        for suffix in ['.npz','.json']:
            target=out/(stem+suffix)
            if not target.exists():shutil.copy2(parent/(stem+suffix),target)
    meta=json.loads((out/'calibration.json').read_text())
    assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['sha256']
    indices=np.load(out/'calibration.npz')['indices']
    for stem in ['calibration_vig','calibration_vig_original','calibration_vig_lowest']:
        m=json.loads((out/(stem+'.json')).read_text());assert m['sha256']==meta['sha256'] and m['stride']==1
        with np.load(out/(stem+'.npz')) as cal:assert np.array_equal(indices,cal['indices'])
    return out,meta


def validate_episode(out,row):
    import h5py
    import numpy as np
    import torch
    from vig_dagger.scoring import w1_per_cell
    with np.load(out/row['trace']) as t,h5py.File(out/row['observations']) as h5:
        steps=row['steps'];assert np.array_equal(t['score_steps'],np.arange(steps))
        assert h5['vig_samples'].shape==(steps,7,32,16,8)
        assert h5['rgb'].shape==(steps,2,256,256,3)
        assert np.array_equal(h5['actions'][:],t['actions'])
        for step,action in enumerate(t['actions']):assert np.array_equal(action,t['replan_chunks'][step//8,step%8])
        for key,pc,cal in [('flow','percentile','calibration'),('vig','vig_percentile','calibration_vig'),('vig_original','vig_original_percentile','calibration_vig_original'),('vig_lowest','vig_lowest_percentile','calibration_vig_lowest')]:
            with np.load(out/(cal+'.npz')) as c:ref=np.sort(c['flow' if key=='flow' else 'vig'])
            assert len(t[key])==steps and np.isfinite(t[key]).all()
            assert np.array_equal(t[pc],100*np.searchsorted(ref,t[key],side='right')/len(ref))
        for step in sorted({0,steps//2,steps-1}):
            samples=torch.from_numpy(h5['vig_samples'][step]);flat=samples.reshape(7,32,-1);mean=samples.mean(dim=2)
            orig=max(float(w1_per_cell(flat[0],flat[i])) for i in range(1,7))
            low=max(float(w1_per_cell(flat[-1],flat[i])) for i in range(6))
            chunk_mean=max(float(w1_per_cell(mean[0],mean[i])) for i in range(1,7))
            assert np.isclose(orig,t['vig_original'][step],rtol=1e-5,atol=1e-7)
            assert np.isclose(low,t['vig_lowest'][step],rtol=1e-5,atol=1e-7)
            assert np.isclose(chunk_mean,t['vig'][step],rtol=1e-5,atol=1e-7)
            assert np.isclose(t['vig_lowest_gap'][step,0],t['vig_original_gap'][step,-1],rtol=1e-6,atol=1e-8)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(out/row['video'])]))['streams'][0]
    assert (probe['width'],probe['height'],int(probe['nb_frames']))==(512,840,steps)


def worker(root,name,preview=False):
    import numpy as np
    import torch
    from vig_dagger.policy import LibraryPolicy,PolicySet
    from vig_dagger.scoring import VIGScorer
    from vig_dagger.insertion_analysis import run_rollout
    from vig_dagger.insertion_env import make_insertion_env
    torch.set_num_threads(2)
    out,meta=prepare_run(root,name)
    if (out/'scoring_complete.json').exists():return
    policy=LibraryPolicy(Path(meta['checkpoint']),'cuda:0')
    refs={}
    for key,stem in [('flow','calibration'),('reference','calibration_vig'),('original_reference','calibration_vig_original'),('lowest_reference','calibration_vig_lowest')]:
        with np.load(out/(stem+'.npz')) as c:refs[key]=np.sort(c['flow' if key=='flow' else 'vig'])
    ctx=dict(policies=PolicySet({1:policy}),scorer=VIGScorer(32,True,True,True,capture_samples=True),**refs)
    cases=json.loads((root/'cases.json').read_text())
    if preview:cases=[c for c in cases if c['case_id']==22]
    rows=json.loads((out/'rollouts.json').read_text()) if (out/'rollouts.json').exists() else []
    done={r['case_id'] for r in rows}
    env=make_insertion_env(render_backend='cuda:0')
    try:
        for case in cases:
            if case['case_id'] in done:continue
            row=run_rollout(policy,env,case,root,out,refs['flow'],'MoE7 | '+name,{},vig_context=ctx,full_video=True,score_stride=1,save_observations=True)
            validate_episode(out,row)
            row['fresh_policy_rollout']=True;rows.append(row);rows.sort(key=lambda r:r['case_id'])
            write_json(out/'rollouts.json',rows)
            write_json(out/'status.json',dict(phase='fresh_rollouts',completed=len(rows),total=50,updated_at=time.time()))
            print('FRESH ROLLOUT',name,case['case_id'],row['success'],row['steps'],flush=True)
    finally:env.close()
    if preview:
        from render_insertion_four_scores import render_row
        row=next(r for r in rows if r['case_id']==22)
        videos=render_row(root,name,row)
        for key in ['video_raw','video_percentile']:
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(root/videos[key])]))['streams'][0]
            assert (probe['width'],probe['height'],int(probe['nb_frames']))==(512,1020,row['steps'])
        write_json(root/'preflight.json',dict(passed=True,fresh_policy_rollout=True,same_sample_vig_verified=True,all_four_calibrations_verified=True,observations_and_samples_retained=True,name=name,**videos,finished_at=time.time()))
    else:
        assert len(rows)==50
        write_json(out/'scoring_complete.json',dict(episodes=50,fresh_policy_rollouts=True,finished_at=time.time()))


def status(root,phase='running'):
    from render_insertion_four_scores import publish
    publish(root)
    runs=list((root/'runs').glob('demos*_epoch*'))
    episodes=sum(len(json.loads((r/'rollouts.json').read_text())) for r in runs if (r/'rollouts.json').exists())
    completed=sum((r/'complete.json').exists() for r in runs)
    calibrated=sum((CALROOT/'runs'/n/'calibration_complete.json').exists() for n in NAMES)
    write_json(root/'status.json',dict(phase=phase,calibrated_checkpoints=calibrated,completed_checkpoints=completed,episodes=episodes,target_episodes=900,target_mp4s=1800,updated_at=time.time()))


def coordinate(root):
    assert json.loads((root/'preflight.json').read_text())['passed']
    prepare(root)
    # Adopt calibration children from the stopped calibration-only coordinator.
    adopted={};pending=queue.Queue()
    for name in NAMES:
        registry=CALROOT/'processes'/f'{name}_calibration.json'
        if registry.exists():
            p=json.loads(registry.read_text())
            if alive(p['pid'],f'--root {CALROOT} --name {name} --calibration-only'):
                assert p['gpu'] not in adopted;adopted[p['gpu']]=name
        fresh_registry=root/'processes'/f'{name}.json'
        if fresh_registry.exists():
            p=json.loads(fresh_registry.read_text())
            if alive(p['pid'],f'--root {root} --worker {name}'):
                assert p['gpu'] not in adopted;adopted[p['gpu']]=name
    ordered=['demos50_epoch300']+[n for n in NAMES if n!='demos50_epoch300']
    for name in ordered:
        if name not in adopted.values():pending.put(name)
    write_json(root/'adopted_calibration_workers.json',adopted)
    provenance=root/'provenance';provenance.mkdir(exist_ok=True)
    for path in ['scripts/run_insertion_four_scores.py','scripts/render_insertion_four_scores.py','scripts/render_insertion_raw_videos.py','src/vig_dagger/scoring.py','src/vig_dagger/insertion_analysis.py','PLUG_INSERTION_PLAN.md']:
        shutil.copy2(REPO/path,provenance/Path(path).name)
    def lane(gpu):
        initial=adopted.get(gpu)
        while True:
            if initial is not None:name=initial;initial=None
            else:
                try:name=pending.get_nowait()
                except queue.Empty:return
            if not (CALROOT/'runs'/name/'calibration_complete.json').exists():
                run_job([sys.executable,'-u','scripts/rescore_insertion_lowest_reference.py','--root',str(CALROOT),'--name',name,'--calibration-only'],CALROOT/'logs'/f'{name}_calibration.log',gpu,CALROOT/'processes'/f'{name}_calibration.json',f'--root {CALROOT} --name {name} --calibration-only')
            assert (CALROOT/'runs'/name/'calibration_complete.json').exists()
            out=root/'runs'/name
            if not (out/'scoring_complete.json').exists():
                run_job([sys.executable,'-u','scripts/run_insertion_four_scores.py','--root',str(root),'--worker',name],root/'logs'/f'{name}.log',gpu,root/'processes'/f'{name}.json',f'--root {root} --worker {name}')
            assert (out/'scoring_complete.json').exists()
            with (root/'logs'/f'{name}_render.log').open('a') as log:
                subprocess.run([sys.executable,'-u','scripts/render_insertion_four_scores.py','--root',str(root),'--name',name],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,gpu) for gpu in range(8)]
        while any(not f.done() for f in futures):
            status(root)
            for f in futures:
                if f.done():f.result()
            time.sleep(15)
        for f in futures:f.result()
    status(root,'plotting')
    subprocess.run([sys.executable,'scripts/render_insertion_four_scores.py','--root',str(root),'--plot'],cwd=REPO,check=True)
    status(root,'complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker');p.add_argument('--preview',action='store_true');a=p.parse_args()
    if a.worker:worker(a.root,a.worker,a.preview)
    else:
        with (a.root/'coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:coordinate(a.root)
            except BaseException as exc:
                write_json(a.root/'status.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
