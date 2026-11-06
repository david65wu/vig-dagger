"""Inspect one completed state: sampled actions, flow losses, and eight outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--policy',default='demos20_dense_full')
p.add_argument('--index',type=int,default=129)
args=p.parse_args()
folder=args.root/args.policy
scores=json.loads((folder/'scores.json').read_text())
row=scores['results'][args.index]
trials=[r for r in json.loads((folder/'rollouts.json').read_text()) if r['index']==args.index]
assert len(trials)==8
arrays=np.load(folder/'action_samples.npz')
samples=arrays['chunks'][args.index,:,0,:3]*1000
ref=arrays['references'][args.index]*1000
threshold=scores['thresholds']['sample_flow_128_p95']
ratios=arrays['losses'][args.index]/threshold
fig,axes=plt.subplots(2,2,figsize=(13,10))
ax=axes[0,0]
points=ax.scatter(samples[:,0],samples[:,1],c=ratios,cmap='coolwarm',vmin=0,vmax=1.5,s=25)
ax.scatter(ref[0],ref[1],marker='x',c='black',s=100,label='Direct-to-B reference')
ax.set(xlabel='First X command (mm)',ylabel='First Y command (mm)',title=f'128 samples; XYZ spread {row["spread_m"]*1000:.3f} mm')
ax.legend();ax.grid(alpha=.2)
fig.colorbar(points,ax=ax,label='Individual chunk flow / ID sample p95')
ax=axes[0,1]
calibration=arrays['losses'][:100].flatten()/threshold
upper=max(1.5,float(np.percentile(calibration,99.5)),float(ratios.max()))
bins=np.linspace(0,upper,50)
ax.hist(calibration,bins=bins,density=True,histtype='step',lw=2,label='Held-out demo samples')
ax.hist(ratios,bins=bins,density=True,alpha=.5,label='Samples at this probe')
ax.axvline(1,color='black',ls='--',label='ID sample p95')
ax.set(xlabel='Individual chunk flow / ID sample p95',ylabel='Density',title='Self-consistency remains in the ID range')
ax.legend(fontsize=9);ax.grid(alpha=.2)
for trial in trials:
    trace=np.load(folder/f"state_{args.index:03d}_trial_{trial['trial']:02d}.npz")
    positions=np.concatenate([trace['states'][:,:3],trace['final_state'][None,:3]])
    axes[1,0].plot(positions[:,1]*1000,(positions[:,0]-.45)*1000,alpha=.65)
    errors=np.linalg.norm(positions-np.array([.45,.15,.35]),axis=1)*1000
    axes[1,1].plot(np.arange(len(errors))/60,errors,alpha=.65)
axes[1,0].axhline(0,color='gray',ls=':')
axes[1,0].scatter(150,0,marker='*',s=160,c='green',label='B')
axes[1,0].scatter(row['position'][1]*1000,(row['position'][0]-.45)*1000,marker='o',s=50,c='black',label='Start')
axes[1,0].set(xlabel='EEF Y position (mm)',ylabel='EEF X offset from line (mm)',title='Eight physical rollout trajectories (XY projection)')
axes[1,0].legend();axes[1,0].grid(alpha=.2)
axes[1,1].axhline(5,color='black',ls='--',label='5 mm success tolerance; must hold 15 steps')
axes[1,1].set(xlabel='Simulation time (seconds)',ylabel='Distance to B (mm)',title=f"Failures: {sum(not r['success'] for r in trials)}/8")
axes[1,1].legend(fontsize=8);axes[1,1].grid(alpha=.2)
fig.suptitle(f"{args.policy}, observation {args.index}: low flow with failed recovery\nMean flow / ID state p95 = {row['flow_ratio']:.2f}; offset = {row['radius']*1000:g} mm, progress = {row['progress']:.2f}")
fig.tight_layout(rect=(0,0,1,.94))
destination=args.root/f'counterexample_{args.policy}_{args.index}.png'
fig.savefig(destination,dpi=160);print(destination)
