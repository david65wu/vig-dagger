"""Reproduce the recorded grasp/lift trace without running the simulator."""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/cpfs02/user/caojianuo/vig-dagger-runs/opendrawer_takeover_sweep_v1')
out=root/'dagger/moe/flow_p95'; dest=root/'results/dagger_audit'
e=next(json.loads(l) for p in (out/'episodes').glob('*.jsonl') for l in p.read_text().splitlines() if json.loads(l)['seed']==1000008)
with h5py.File(out/'attempts/worker_00.h5','r') as f:
 g=f['seed_1000008'];tcp=g['tcp_pose'][:,:3];obj=g['charger_pose'][:,:3];gap=g['qpos'][:,-2:].sum(axis=1);hold=g['holding_charger'][:];control=g['controller'][:]
with h5py.File(out/e['file'],'r') as f:
 g=f[f"data/demo_{e['traj']}"];actions=g['actions'][:];strategy=json.loads(g.attrs['expert_strategies'])[-1]
start=334;end=350
summary=dict(variant='moe/flow_p95',seed=1000008,takeover_step=215,lift_action_start=start,
             detected_grasp_loss_step=337,termination_step=end,
             object_max_rise_from_pregrasp_mm=float((obj[322:,2].max()-obj[322,2])*1000),
             hand_lift_mm=float((tcp[end,2]-tcp[start,2])*1000),
             object_horizontal_displacement_mm=float(np.linalg.norm(obj[end,:2]-obj[start,:2])*1000),
             max_hand_step_mm=float(np.linalg.norm(np.diff(tcp[start:end+1],axis=0),axis=1).max()*1000),
             close_command_through_lift=bool(np.all(actions[start:end,6]==-1)),
             expert_through_lift=bool(np.all(control[start:end]==1)),
             strategy={k:strategy[k] for k in ['speed_transit','speed_careful','dwell_steps','lift_dx','lift_dy','charger_grasp_z']})
(dest/'lift_failure_1000008.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True,layout='constrained')
t=np.arange(310,351);baseline=obj[322,2]
axs[0].plot(t,(tcp[t,2]-baseline)*1000,label='Hand TCP');axs[0].plot(t,(obj[t,2]-baseline)*1000,label='Charger root');axs[0].set_ylabel('Height above initial\ncharger root (mm)');axs[0].legend()
axs[1].plot(t,gap[t]*1000,label='Finger gap');axs[1].axhline(25,color='red',ls='--',label='Holding detector lower bound');axs[1].set_ylabel('Finger gap (mm)');axs[1].legend()
axs[2].step(t,hold[t].astype(int),where='post',label='Holding detector');axs[2].set_ylabel('Detected holding (0/1)');axs[2].set_xlabel('Recorded observation / action step (60 Hz)');axs[2].set_yticks([0,1])
for ax in axs:
 ax.axvline(334,color='gray',ls=':',label='Lift starts');ax.axvline(337,color='red',ls=':',label='Grasp loss detected');ax.grid(alpha=.2)
axs[0].set_title('MoE flow p95 · attempt 1000008\nExpert controls the entire displayed interval; lift starts at 334, grasp lost at 337')
fig.savefig(dest/'lift_failure_1000008.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
