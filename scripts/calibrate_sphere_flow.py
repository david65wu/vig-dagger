"""Calibrate generated-action flow loss on each sphere policy's own training demos."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np

from run_sphere_rollouts import NAMES,write


def prepare(root, source=None):
    source=source or root.parent/'toy_sphere_20_demos_v1/dataset/toy_reach'
    shards=json.loads((source/'demo_manifest.json').read_text())['shards']
    def read(shard):
        with h5py.File(source/shard['hdf5']) as f:
            g=next(iter(f['data'].values()));length=len(g['actions'])
            frames=np.linspace(0,length-16,20).astype(int)
            assert len(np.unique(frames))==20
            states=g['obs/state'][frames]
        cameras=[]
        for camera in ['third_person','wrist']:
            cap=cv2.VideoCapture(str(source/shard['videos'][camera]['mp4']));wanted=set(frames);saved={};index=0
            try:
                while wanted:
                    ok,bgr=cap.read();assert ok,(camera,shard['shard_id'],index)
                    if index in wanted:saved[index]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);wanted.remove(index)
                    index+=1
            finally:cap.release()
            cameras.append(np.stack([saved[i] for i in frames]))
        return states,np.stack(cameras,axis=1),[dict(demo=shard['shard_id'],frame=int(i)) for i in frames]
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:data=list(pool.map(read,shards))
    states=np.concatenate([d[0] for d in data]);images=np.concatenate([d[1] for d in data]);records=[r for d in data for r in d[2]]
    assert states.shape==(400,12) and images.shape==(400,2,256,256,3)
    out=root/'flow_calibration';out.mkdir(exist_ok=True)
    with h5py.File(out/'training_observations.hdf5','w') as f:
        f.create_dataset('states',data=states);f.create_dataset('images',data=images,compression='gzip',compression_opts=1)
        f.attrs['records']=json.dumps(records)
    write(out/'protocol.json',dict(source=str(source),frames_per_demo=20,frames_5_demos=100,frames_20_demos=400,
        sampling='20 evenly spaced eligible frames per demo; includes first frame and last 16-action training-window start',
        weighting='Equal number of frames per demo',images='Decoded original training MP4s; full resolution before normal inference preprocessing',
        score='Mean flow-matching loss of 32 policy-generated 16-step action chunks, each with 64 noise/time draws',
        statistic='Per-policy empirical median and 95th percentile of observation-level mean losses',
        caveat='In-sample training calibration, not held-out OOD validation; training p95 is a reference, not a guaranteed OOD boundary'))


def score(root,name,gpu,training_root=None,checkpoint='epoch-100.pt'):
    import torch
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_flow_search import sample_and_score
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cv2.setNumThreads(1)
    training=(training_root or root.parent/'toy_sphere_training_v1')/'policies'/name
    normalization=json.loads((training/'normalization.json').read_text())
    selected={int(s[1]) for s in normalization['selection']}
    assert selected==set(range(5 if name.startswith('demos5_') else 20))
    policy=LibraryPolicy(training/checkpoint,f'cuda:{gpu}')
    rows=[];losses=[];started=time.time()
    with h5py.File(root/'flow_calibration/training_observations.hdf5') as f:
        records=json.loads(f.attrs['records'])
        for index,r in enumerate(records):
            if r['demo'] not in selected:continue
            state=torch.tensor(f['states'][index],device=policy.device)[None]
            images=torch.tensor(f['images'][index],device=policy.device).permute(0,3,1,2)[None]
            values,_,flow,_=sample_and_score(policy,state,images,seed=131000+index,samples=32,draws=64)
            assert np.isfinite(flow).all();rows.append(dict(r,flow=values['flow_mean']));losses.append(flow)
            if len(rows)%20==0:
                write(root/'flow_calibration'/f'{name}_progress.json',dict(completed=len(rows),total=20*len(selected),elapsed_seconds=time.time()-started))
    scores=np.array([r['flow'] for r in rows]);median,p95=np.percentile(scores,[50,95])
    np.savez_compressed(root/'flow_calibration'/f'{name}.npz',flow=scores,chunk_losses=np.array(losses))
    write(root/'flow_calibration'/f'{name}.json',dict(policy=name,checkpoint=str(training/checkpoint),training_demos=len(selected),frames=len(rows),
        samples=32,draws=64,median=float(median),p95=float(p95),results=rows))


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8));summary=[]
    for ax,name in zip(axes.flat,NAMES):
        d=json.loads((root/'flow_calibration'/f'{name}.json').read_text());flow=[r['flow'] for r in d['results']]
        ax.hist(flow,bins=20,color='#4c88b5',alpha=.75)
        ax.axvline(d['p95'],color='#934c9f',ls='--',lw=2,label=f'train p95 = {d["p95"]:.4f}')
        ax.set(title=name.replace('_',' '),xlabel='Generated-action flow loss',ylabel='Training observations')
        ax.legend(fontsize=9);summary.append({k:v for k,v in d.items() if k!='results'})
    fig.suptitle('Training-demo flow calibration: same scoring recipe as rollout videos\n20 frames per demo; 100 observations for five-demo policies, 400 for twenty-demo policies')
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(root/'flow_calibration/training_flow_calibration.png',dpi=160);plt.close(fig)
    write(root/'flow_calibration/summary.json',summary)


def run(root):
    (root/'flow_calibration').mkdir(exist_ok=True)
    write(root/'flow_calibration/status.json',dict(state='preparing'))
    prepare(root);processes=[]
    for gpu,name in enumerate(NAMES):
        with (root/'logs'/f'calibrate_{name}.log').open('w') as log:
            p=subprocess.Popen([sys.executable,__file__,'score','--root',str(root),'--name',name,'--gpu',str(gpu)],
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,p))
    write(root/'flow_calibration/processes.json',[dict(name=n,pid=p.pid) for n,p in processes])
    write(root/'flow_calibration/status.json',dict(state='scoring'))
    for name,p in processes:
        if p.wait():raise RuntimeError(f'Calibration failed: {name}')
    report(root);write(root/'flow_calibration/status.json',dict(state='complete',policies=6,observations=1500))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','score','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--name');p.add_argument('--gpu',type=int)
    p.add_argument('--training',type=Path);p.add_argument('--checkpoint',default='epoch-100.pt')
    a=p.parse_args()
    if a.mode=='score':score(a.root,a.name,a.gpu,a.training,a.checkpoint)
    elif a.training is not None or a.checkpoint!='epoch-100.pt':
        p.error('Explicit training/checkpoint options currently require score mode')
    else:globals()[a.mode](a.root)
