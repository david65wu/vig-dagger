"""Calibrate and evaluate VIG after averaging each sampled action chunk."""
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
import tempfile
import time

from run_insertion_experiment import REPO,run_job,write_json

BASE=Path('/cpfs02/user/caojianuo/vig-dagger-runs')
OLD=BASE/'charger_insertion_moe7_every_frame_v1'
TRAINING=BASE/'charger_insertion_moe7_v1'
EPOCHS=[10,25,50,100,200,300]


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((OLD/'protocol.json').read_text())
    protocol.update(vig_temporal_average=True,policy_label='7-level MoE | VIG after 16-step chunk mean | scores every frame',
        vig_metric='Average each normalized 16-step sample chunk to 8 dimensions; per degraded resolution compute mean per-dimension 1D Wasserstein over 32 samples; maximum over six degraded resolutions',
        prior_evaluation=str(OLD),vig_calibration='New reference on every pre-action training frame, specific to checkpoint and chunk-mean VIG')
    write_json(root/'protocol.json',protocol);shutil.copy2(OLD/'cases.json',root/'cases.json')
    for row in json.loads((OLD/'cases.json').read_text()):
        src=OLD/row['state_file'];dst=root/row['state_file'];dst.parent.mkdir(exist_ok=True)
        if not dst.exists():os.link(src,dst)
    for count in [50,200,500]:
        for epoch in EPOCHS:
            name=f'demos{count}_epoch{epoch:03d}';src=OLD/'runs'/name;dst=root/'runs'/name;dst.mkdir(parents=True,exist_ok=True)
            meta=json.loads((src/'calibration.json').read_text())
            checkpoint=TRAINING/'policies'/f'demos{count}_moe_7level'/f'epoch-{epoch:03d}.pt'
            assert meta['stride']==1 and meta['sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for filename in ['calibration.npz','calibration.json']:
                if not (dst/filename).exists():shutil.copy2(src/filename,dst/filename)
            write_json(dst/'flow_reference_source.json',dict(source=str(src),checkpoint_sha256=meta['sha256']))
    if not (root/'index.html').exists():
        links=''.join(f'<li><a href="runs/demos{n}_epoch{e:03d}/index.html">{n} demos / epoch {e}</a></li>' for n in [50,200,500] for e in EPOCHS)
        (root/'index.html').write_text('<html><meta charset="utf-8"><body><h1>MoE: VIG after averaging 16-step chunks</h1><p>In progress. New VIG calibration and every-frame scores. Each checkpoint gallery will contain all 50 videos.</p><ul>'+links+'</ul></body></html>')


def preflight(root):
    import numpy as np
    import torch
    import h5py
    from vig_dagger.scoring import VIGScorer
    from vig_dagger.policy import LibraryPolicy,PolicySet
    from vig_dagger.insertion_analysis import score_vig,inputs,run_rollout
    from vig_dagger.insertion_env import make_insertion_env
    torch.set_num_threads(2)
    class Fake:
        device=torch.device('cpu');horizon=16;action_dim=8;factors=(1,2)
        def __init__(self,samples):self.samples=samples
        def vig_samples(self,*args):return self.samples
    samples=torch.zeros(2,2,16,8);samples[1,:,0:8]=1;samples[1,:,8:]=-1
    state=torch.zeros(1,9);images=torch.zeros(1,2,3,2,2)
    assert VIGScorer(2,True)(Fake(samples),state,images).value==0
    assert VIGScorer(2,False)(Fake(samples),state,images).value==1
    samples=torch.zeros(2,2,16,8);samples[1,0]=-1;samples[1,1]=1
    assert VIGScorer(2,True)(Fake(samples),state,images).value==1
    samples=torch.zeros(2,2,16,8);samples[1,:,:,0]=8
    assert VIGScorer(2,True)(Fake(samples),state,images).value==1
    policy=LibraryPolicy(TRAINING/'policies/demos50_moe_7level/epoch-300.pt','cuda:0')
    context=dict(policies=PolicySet({1:policy}),scorer=VIGScorer(32,True))
    demo=json.loads((TRAINING/'demo_order.json').read_text())[0];values=[]
    with h5py.File(Path(demo['source']).with_suffix('.h5')) as h5:
        group=h5[f'traj_{demo["episode_id"]}']
        for frame in range(16):
            st,im=inputs(policy,group['obs/agent/qpos'][frame],np.stack([group[f'obs/sensor_data/{c}/rgb'][frame] for c in policy.cameras]))
            value,gaps=score_vig(context,st,im,80000000+demo['source_index']*1000+frame)
            assert value==max(gaps);values.append(value)
    context['reference']=np.sort(values)
    case=json.loads((root/'cases.json').read_text())[0]
    out=Path(tempfile.mkdtemp(prefix='mean_chunk_vig_smoke_'));env=make_insertion_env(render_backend='cuda:0')
    reference=np.sort(np.load(root/'runs/demos50_epoch300/calibration.npz')['flow'])
    try:row=run_rollout(policy,env,case,root,out,reference,'CHUNK MEAN TEST',{},vig_context=context,full_video=True,score_stride=1,max_steps=16)
    finally:env.close()
    trace=np.load(out/row['trace'])
    assert row['vig_temporal_average'] and len(trace['vig'])==16
    assert np.array_equal(trace['score_steps'],np.arange(16)) and np.array_equal(trace['replan_steps'],[0,8])
    assert np.allclose(trace['vig'],trace['vig_gap'].max(axis=1))
    assert np.allclose(trace['vig_percentile'],100*np.searchsorted(context['reference'],trace['vig'],side='right')/16)
    for step,action in enumerate(trace['actions']):assert np.array_equal(action,trace['replan_chunks'][step//8,step%8])
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out/row['video']),'-f','null','-'],capture_output=True,check=True)
    write_json(root/'preflight.json',dict(passed=True,temporal_axis_verified=True,sample_distribution_preserved=True,action_dimensions_preserved=True,
        frames=16,scores=16,replans=2,rng_preservation_verified=context['rng_preservation_verified'],test_artifacts=str(out),
        calibration_note='Smoke test uses 16 new training-frame VIG scores; production recalibrates on ALL own training frames',finished_at=time.time()))


def coordinate(root):
    prepare(root)
    write_json(root/'pipeline.json',dict(phase='preflight',pid=os.getpid(),updated_at=time.time()))
    if not (root/'preflight.json').exists():run_job([sys.executable,'-u','scripts/run_insertion_mean_chunk_vig.py','--root',str(root),'--preflight'],root/'preflight.log',0,root/'preflight_process.json','--preflight')
    assert json.loads((root/'preflight.json').read_text())['passed']
    pending=queue.Queue()
    for count in [500,200,50]:
        for epoch in EPOCHS:pending.put((count,epoch))
    write_json(root/'pipeline.json',dict(phase='calibration_and_evaluation',pid=os.getpid(),gpus=list(range(8)),checkpoints=18,rollouts=900,updated_at=time.time()))
    def lane(gpu):
        while True:
            try:count,epoch=pending.get_nowait()
            except queue.Empty:return
            name=f'demos{count}_epoch{epoch:03d}';out=root/'runs'/name
            if not (out/'complete.json').exists():
                run_job([sys.executable,'-u','-m','vig_dagger.insertion_analysis','worker','--root',str(root),'--training',str(TRAINING),'--count',str(count),'--epoch',str(epoch)],root/'logs'/f'{name}.log',gpu,root/'processes'/f'{name}.json',f'--root {root} --training {TRAINING} --count {count} --epoch {epoch}')
            assert (out/'complete.json').exists()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,g) for g in range(8)]
        for future in futures:future.result()
    write_json(root/'pipeline.json',dict(phase='plotting',updated_at=time.time()))
    subprocess.run([sys.executable,'scripts/plot_insertion_analysis.py','--root',str(root)],cwd=REPO,check=True)
    subprocess.run([sys.executable,'scripts/compare_insertion_variants.py','--dense',str(BASE/'charger_insertion_dense_every_frame_v1'),'--moe',str(root)],cwd=REPO,check=True)
    write_json(root/'pipeline.json',dict(phase='complete',checkpoints=18,rollouts=900,finished_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if a.preflight:preflight(a.root)
    else:
        with (a.root/'coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:coordinate(a.root)
            except BaseException as exc:
                write_json(a.root/'pipeline.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
