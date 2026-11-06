"""Diagnostic flow along a recorded rollout, using decoded review-video RGB."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from vig_dagger.policy import LibraryPolicy
from vig_dagger.toy_flow_search import sample_and_score

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--policy',default='demos20_dense_full')
p.add_argument('--index',type=int,default=129)
args=p.parse_args()
folder=args.root/args.policy
run=Path('/cpfs02/user/caojianuo/vig-dagger-runs/toy_reach_training_v1/policies')/args.policy
scores=json.loads((folder/'scores.json').read_text())
threshold=scores['thresholds']['state_flow_p95']
trace=np.load(folder/f'state_{args.index:03d}_trial_00.npz')
path=folder/'videos'/f'state_{args.index:03d}_trial_00.mp4'
# Videos contain every second simulation step. Select every fourth video frame.
raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-vf',
    r'select=not(mod(n\,4))','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'])
panels=np.frombuffer(raw,dtype=np.uint8).reshape(-1,284,512,3)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
policy=LibraryPolicy(run/'latest.pt','cuda:0')
rows=[]
with torch.inference_mode():
    for frame,panel in enumerate(panels):
        step=frame*8
        if step>=len(trace['states']):break
        if step>160 and step%32:continue
        rgb=np.stack([panel[28:,:256],panel[28:,256:]])
        state=trace['states'][step]
        s=torch.tensor(state,device=policy.device)[None]
        im=torch.tensor(rgb,device=policy.device).permute(0,3,1,2)[None]
        values,_,_,_=sample_and_score(policy,s,im,seed=41000+step)
        row=dict(step=step,seconds=step/60,goal_error_m=float(np.linalg.norm(state[:3]-np.array([.45,.15,.35]))),
                 flow_mean=values['flow_mean'],flow_ratio=values['flow_mean']/threshold,spread_m=values['spread_m'])
        rows.append(row)
        print(json.dumps(row),flush=True)
result=dict(policy=args.policy,index=args.index,trial=0,image_source='Decoded H.264 review frames; diagnostic, not raw scoring inputs.',
            draws=128,samples=128,threshold=threshold,results=rows)
dest=args.root/f'trajectory_flow_{args.policy}_{args.index}'
dest.with_suffix('.json').write_text(json.dumps(result,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
times=[r['seconds'] for r in rows]
axes[0].plot(times,[r['flow_ratio'] for r in rows],marker='.',color='purple')
axes[0].axhline(1,color='black',ls='--',label='Held-out ID p95')
axes[0].set(ylabel='Mean flow / ID state p95');axes[0].legend();axes[0].grid(alpha=.2)
axes[1].plot(times,[r['goal_error_m']*1000 for r in rows],marker='.')
axes[1].axhline(5,color='black',ls='--',label='Goal tolerance (5 mm)')
axes[1].set(xlabel='Simulation time (seconds)',ylabel='Distance to B (mm)');axes[1].legend();axes[1].grid(alpha=.2)
fig.suptitle(f'{args.policy}, state {args.index}, trial 0: flow through the failed rollout\nDiagnostic scores from decoded review-video frames')
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(dest.with_suffix('.png'),dpi=160)
early=[r for r in rows if r['seconds']<=3]
axes[0].set(xlim=(0,3),ylim=(0,max(1.2,max(r['flow_ratio'] for r in early)*1.15)))
axes[1].set(ylim=(0,max(r['goal_error_m']*1000 for r in early)*1.15))
fig.savefig(dest.with_name(dest.name+'_early').with_suffix('.png'),dpi=160)
