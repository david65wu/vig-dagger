"""Sphere comparison with only flow loss and sampled action distributions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np

NAMES=[f'demos{n}_{v}' for n in (5,20) for v in ('dense_full','moe_7level','dense_7level')]
LABELS=['ID: recorded demo','OOD: 3 cm','OOD: 8 cm']
COLORS=['#2878b5','#ed9121','#c73840']


def write(path, data):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def score(args):
    import torch
    from vig_dagger.policy import LibraryPolicy
    from vig_dagger.toy_flow_search import sample_and_score
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=args.root/args.name;out.mkdir(exist_ok=True)
    policy=LibraryPolicy(args.training/'policies'/args.name/args.checkpoint,'cuda:0')
    rows,chunks,losses=[],[],[]
    started=time.time()
    with torch.inference_mode(),h5py.File(args.root/'observations.hdf5') as f:
        records=json.loads(f.attrs['records'])
        for index,record in enumerate(records):
            state=torch.tensor(f['states'][index],device=policy.device)[None]
            rgb=torch.tensor(f['images'][index],device=policy.device).permute(0,3,1,2)[None]
            # Matched action noise and scoring draws for each ID/OOD anchor family.
            seed=41000+record['demo']*100+int(record['progress']*100)
            values,raw,flow,_=sample_and_score(policy,state,rgb,seed,samples=128,draws=128)
            assert np.isfinite(raw).all() and np.isfinite(flow).all()
            rows.append(dict(record,index=index,flow_loss=values['flow_mean']))
            chunks.append(raw);losses.append(flow)
            if (index+1)%10==0:
                progress=dict(done=index+1,total=len(records),elapsed_seconds=time.time()-started)
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        np.savez_compressed(out/'samples.npz',chunks=np.array(chunks),flow_losses=np.array(losses),
                            states=f['states'][:],references=f['references'][:])
    write(out/'scores.json',dict(checkpoint=policy.checkpoint,samples=128,draws=128,results=rows))


def group_indices(rows):
    return [[r['index'] for r in rows if r['kind']=='id'],
            [r['index'] for r in rows if r['kind']=='ood' and r['radius_m']==.03],
            [r['index'] for r in rows if r['kind']=='ood' and r['radius_m']==.08]]


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    data={name:json.loads((args.root/name/'scores.json').read_text()) for name in NAMES}
    arrays={name:np.load(args.root/name/'samples.npz') for name in NAMES}
    summary=[]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,name in zip(axes.flat,NAMES):
        rows=data[name]['results'];groups=group_indices(rows)
        values=[[rows[i]['flow_loss'] for i in ids] for ids in groups]
        recorded=[r['flow_loss'] for r in rows if r['kind']=='recorded_id']
        shown=[recorded]+values
        bp=ax.boxplot(shown,positions=[0,1,2,3],widths=.45,patch_artist=True,showfliers=False,
                      medianprops=dict(color='black',linewidth=1.5))
        rng=np.random.default_rng(0)
        for j,(val,color,box) in enumerate(zip(shown,['#6f96ad']+COLORS,bp['boxes'])):
            box.set_facecolor(color);box.set_alpha(.25)
            ax.scatter(j+rng.uniform(-.15,.15,len(val)),val,color=color,s=13,alpha=.7)
        ax.set(xticks=[0,1,2,3],xticklabels=['Recorded\nID','On-path\ncontrol','OOD\n3 cm','OOD\n8 cm'],ylabel='Flow loss (mean over sampled chunks)',
               title=name.replace('_',' '))
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        summary.append(dict(name=name,median_flow=dict(zip(('id','ood_3cm','ood_8cm'),map(lambda v:float(np.median(v)),values))),
                            counts=dict(zip(('id','ood_3cm','ood_8cm'),map(len,values))),
                            recorded_id_median_flow=float(np.median(recorded))))
    fig.suptitle('Sphere: flow loss at ID versus OOD observations\nRecorded ID = actual demo frames; on-path control = recreated poses; OOD = sideways offsets, same goals.\nEach dot = one observation; box = middle 50%; line = median. Full-resolution images for every policy.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(args.root/'flow_id_vs_ood.png',dpi=170);plt.close(fig)

    # Examples are chosen by geometry before scoring: midpoint, +u, demos 0..4.
    examples=args.root/'examples';examples.mkdir(exist_ok=True)
    for demo in range(5):
        rows=data[NAMES[0]]['results']
        ids=[next(r['index'] for r in rows if r['demo']==demo and r['progress']==.5
                  and r['direction']==0 and r['radius_m']==radius
                  and r['kind']==('recorded_id' if radius==0 else 'ood'))
             for radius in (0.,.03,.08)]
        along=np.array(rows[ids[0]]['along']);u=np.array(rows[ids[0]]['transverse_u'])
        projected={name:[arrays[name]['chunks'][i,:,0,:3]*1000 @ np.stack((along,u)).T for i in ids] for name in NAMES}
        positions=arrays[NAMES[0]]['states'][ids,:3]
        goals=arrays[NAMES[0]]['states'][ids,9:12]
        delta=goals-positions
        reference_xyz=delta*np.minimum(8/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-9),1000)
        ref=reference_xyz @ np.stack((along,u)).T
        all_points=np.concatenate([x for sets in projected.values() for x in sets]+[ref])
        lo=all_points.min(axis=0);hi=all_points.max(axis=0)
        center=(lo+hi)/2;half=max(float((hi-lo).max())*.58,.5)
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,NAMES):
            for j,(cloud,color,label) in enumerate(zip(projected[name],COLORS,LABELS)):
                flow=data[name]['results'][ids[j]]['flow_loss']
                ax.scatter(*cloud.T,color=color,s=14,alpha=.55,label=label)
                ax.scatter(*ref[j],color=color,marker='x',s=85,linewidths=2)
            flows=' / '.join(f'{data[name]["results"][i]["flow_loss"]:.3f}' for i in ids)
            ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
                   xlabel='Along A-to-B command (mm)',ylabel='Sideways command (mm)',
                   title=name.replace('_',' ')+f'\nFlow (ID / 3 cm / 8 cm): {flows}')
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        handles,_=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,LABELS,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3)
        fig.suptitle(f'Sphere action samples: demo {demo}, halfway to B, same goal\nDots = first command from 128 sampled chunks, projected onto path/offset plane. Crosses = direct-to-goal command.',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.88))
        target=examples/f'demo_{demo:03d}_action_distributions.png'
        fig.savefig(target,dpi=170);plt.close(fig)
        if demo==0:shutil.copy2(target,args.root/'action_distributions.png')

        # All three Cartesian projections preserve the out-of-plane component.
        for name in NAMES:
            fig,axes=plt.subplots(3,3,figsize=(11,10))
            xyz=[arrays[name]['chunks'][i,:,0,:3]*1000 for i in ids]
            refs=reference_xyz
            bounds=np.concatenate(xyz+[refs]);lower=bounds.min(0);upper=bounds.max(0)
            pad=np.maximum((upper-lower)*.1,.2)
            for row,(p,q) in enumerate(((0,1),(0,2),(1,2))):
                for col,(cloud,color) in enumerate(zip(xyz,COLORS)):
                    ax=axes[row,col];ax.scatter(cloud[:,p],cloud[:,q],s=14,color=color,alpha=.65)
                    ax.scatter(refs[col,p],refs[col,q],marker='x',color='black',s=65)
                    ax.set(xlabel=f'{"XYZ"[p]} command (mm)',ylabel=f'{"XYZ"[q]} command (mm)',
                           xlim=(lower[p]-pad[p],upper[p]+pad[p]),ylim=(lower[q]-pad[q],upper[q]+pad[q]))
                    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
                    if row==0:ax.set_title(f'{LABELS[col]}\nflow={data[name]["results"][ids[col]]["flow_loss"]:.4g}')
            fig.suptitle(f'{name} | demo {demo}, midpoint | 128 samples per observation\nFirst-command projections; common limits within each row; black cross = direct-to-goal command',fontsize=11)
            fig.tight_layout(rect=(0,0,1,.93));fig.savefig(examples/f'demo_{demo:03d}_{name}_xyz.png',dpi=140);plt.close(fig)
    # Small input review: both cameras for the main three matched observations.
    rows=data[NAMES[0]]['results']
    ids=[next(r['index'] for r in rows if r['demo']==0 and r['progress']==.5 and r['direction']==0
              and r['radius_m']==radius and r['kind']==('recorded_id' if radius==0 else 'ood')) for radius in (0.,.03,.08)]
    panel=Image.new('RGB',(512,3*284),'white');draw=ImageDraw.Draw(panel)
    with h5py.File(args.root/'observations.hdf5') as f:
        for j,i in enumerate(ids):
            draw.text((8,j*284+7),LABELS[j]+' | third person / wrist',fill='black')
            for c in range(2):panel.paste(Image.fromarray(f['images'][i,c]),(c*256,j*284+28))
    panel.save(args.root/'example_observations.png')
    write(args.root/'summary.json',summary)
    for a in arrays.values():a.close()
    print(json.dumps(summary,indent=2))


def run(args):
    repo=Path(__file__).resolve().parents[1]
    write(args.root/'protocol.json',dict(training=str(args.training),checkpoints=args.checkpoint,
        id='15 physically placed on-path observations: first five training demos at 25%, 50%, 75% progress',
        ood='Same A/B; EEF offset 3 or 8 cm along four perpendicular directions; rendered physical observations',
        baseline_check='15 exact recorded training observations are the recorded ID reference; 15 recreated on-path poses control for physical placement effects',
        scope='Spatial off-path OOD conditional on familiar goals, not unseen-goal generalization; ID is training-supported, not held-out',
        flow='Mean linear flow-matching MSE of 128 policy-generated 16-step chunks, each scored with 128 fresh noise/time draws; normalized 7D actions',
        action_plot='First translation command in mm from each of 128 chunks; recorded ID versus physically displaced OOD; fixed geometry examples, never selected by score',
        examples='Demo 0..4, progress .5, direction +u; main figure uses demo 0',image_resolution='Full, factor 1 for every policy',
        metrics=['flow_loss','action_samples']))
    write(args.root/'pipeline.json',dict(state='capturing',pid=os.getpid()))
    started=time.time()
    while not (args.root/'capture_done.json').exists():
        capture_pid=json.loads((args.root/'capture_process.json').read_text())['pid']
        if not Path(f'/proc/{capture_pid}').exists():raise RuntimeError('Capture exited before completion; see logs/capture.log')
        if time.time()-started>1800:raise TimeoutError('Capture exceeded 30 minutes')
        time.sleep(5)
    processes=[]
    for gpu,name in enumerate(NAMES):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
        with (args.root/'logs'/f'{name}.log').open('w') as log:
            p=subprocess.Popen([sys.executable,__file__,'score','--root',str(args.root),'--training',str(args.training),'--name',name,
                                '--checkpoint',args.checkpoint],
                               cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,p))
    write(args.root/'processes.json',[dict(name=n,pid=p.pid,gpu=i) for i,(n,p) in enumerate(processes)])
    write(args.root/'pipeline.json',dict(state='scoring',pid=os.getpid()))
    for name,p in processes:
        if p.wait():raise RuntimeError(f'{name} failed; see its log')
    write(args.root/'pipeline.json',dict(state='plotting',pid=os.getpid()))
    report(args)
    write(args.root/'pipeline.json',dict(state='complete',pid=os.getpid(),elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','score','report'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training',type=Path,default=Path('/cpfs02/user/caojianuo/vig-dagger-runs/toy_sphere_training_v1'))
    p.add_argument('--name',choices=NAMES)
    p.add_argument('--checkpoint',default='epoch-100.pt')
    args=p.parse_args()
    try:globals()[args.mode](args)
    except BaseException as exc:
        if args.mode=='run':write(args.root/'pipeline.json',dict(state='failed',error=repr(exc)))
        traceback.print_exc();raise
