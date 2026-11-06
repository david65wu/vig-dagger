"""Prepare, launch detached sphere rollouts, and summarize only success counts."""
import argparse
import csv
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
KINDS=['familiar_a_to_b','unseen_a_to_b','midpoint_id','midpoint_3cm','midpoint_8cm']


def write(path,data):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def prepare(root):
    from vig_dagger.toy_reach_env.sphere_sampling import sample_sphere_pair
    source=root.parent/'toy_sphere_simple_eval_v1'
    with h5py.File(source/'observations.hdf5') as f:records=json.loads(f.attrs['records'])
    cases=[]
    for demo in range(5):
        for radius,kind in [(0.,'midpoint_id'),(.03,'midpoint_3cm'),(.08,'midpoint_8cm')]:
            r=next(r for r in records if r['demo']==demo and r['progress']==.5 and r['direction']==0
                   and r['radius_m']==radius and r['kind']==('recorded_id' if radius==0 else 'ood'))
            cases.append(dict(kind=kind,demo=demo,frame=r['frame'],point_a=r['point_a'],point_b=r['point_b'],
                              start=r['anchor'] if radius==0 else r['requested_position'],reset_seed=20000+demo))
    for demo in range(5):
        r=next(r for r in records if r['demo']==demo)
        cases.append(dict(kind='familiar_a_to_b',demo=demo,point_a=r['point_a'],point_b=r['point_b'],
                          start=r['point_a'],reset_seed=100+demo))
    for seed in range(6000,6005):
        a,b=sample_sphere_pair((.45,0,.35),.15,.10,seed)
        cases.append(dict(kind='unseen_a_to_b',point_a=a.tolist(),point_b=b.tolist(),start=a.tolist(),reset_seed=seed))
    for i,c in enumerate(cases):c.update(case_id=i,policy_seed=81000+i)
    write(root/'cases.json',cases)
    write(root/'protocol.json',dict(policies=NAMES,checkpoint='epoch-100.pt',cases_per_policy=25,total_rollouts=150,
        groups=KINDS,trials_per_start=1,success='EEF distance to B < 5 mm for 15 consecutive control steps',
        max_steps=1200,control_hz=60,chunk_horizon=16,execute_steps_before_replan=8,image_factor=1,
        familiar_pairs='First five training pairs, common to every model',unseen_pair_seeds=list(range(6000,6005)),
        midpoint_cases='Same five midpoint examples and positive-u offsets as static action plots',
        midpoint_id_initialization='Recorded joint positions and physical velocities, zero desired joint velocity, refreshed cameras; state agreement asserted',
        caveat='Five starts per group, one stochastic rollout each: descriptive counts, not precise population rates',
        aborts='nonfinite action, translation command norm > 5 cm, rotation command norm > .2 rad, or workspace exit'))


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=json.loads((root/'cases.json').read_text())
    kinds=KINDS+(['outside_sphere_a_to_b'] if any(c['kind']=='outside_sphere_a_to_b' for c in cases) else [])
    summaries=[];table=np.zeros((6,len(kinds)))
    for i,name in enumerate(NAMES):
        rows=json.loads((root/name/'rollouts.json').read_text())
        assert len(rows)==len(cases) and len({r['case_id'] for r in rows})==len(cases)
        grouped={}
        for j,kind in enumerate(kinds):
            group=[r for r in rows if r['kind']==kind];assert len(group)==5
            successes=sum(r['success'] for r in group);table[i,j]=successes/5
            grouped[kind]=dict(successes=successes,episodes=5,success_rate=successes/5)
        summaries.append(dict(policy=name,groups=grouped))
    write(root/'summary.json',summaries)
    with (root/'success_rates.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['policy','group','successes','episodes','success_rate'])
        for row in summaries:
            for kind,s in row['groups'].items():writer.writerow([row['policy'],kind,s['successes'],s['episodes'],s['success_rate']])
    fig,ax=plt.subplots(figsize=(13 if len(kinds)==6 else 11,5))
    ax.imshow(table,cmap='YlGn',vmin=0,vmax=1,aspect='auto')
    labels=['Familiar A→B','Unseen A→B','Midpoint ID','Midpoint +3 cm','Midpoint +8 cm']
    if len(kinds)==6:labels.append('Both endpoints\noutside sphere')
    ax.set(xticks=range(len(kinds)),xticklabels=labels,
           yticks=range(6),yticklabels=[n.replace('_',' ') for n in NAMES])
    for i in range(6):
        for j in range(len(kinds)):ax.text(j,i,f'{int(round(table[i,j]*5))}/5 ({table[i,j]*100:.0f}%)',ha='center',va='center',color='white' if table[i,j]>.7 else 'black')
    ax.set_title('Sphere closed-loop success rates\nSuccess: within 5 mm of B for 15 steps; 20-second limit. One rollout per start.',pad=14)
    fig.tight_layout();fig.savefig(root/'success_rates.png',dpi=170);plt.close(fig)
    print(json.dumps(summaries,indent=2),flush=True)


def run(root):
    repo=Path(__file__).resolve().parents[1];root.mkdir(exist_ok=True)
    (root/'logs').mkdir(exist_ok=True);(root/'source_snapshot').mkdir(exist_ok=True)
    if not (root/'cases.json').exists():prepare(root)
    for rel in ['scripts/rollout_sphere_simple.py','scripts/run_sphere_rollouts.py','scripts/toy_flow_search_sim.py',
                'src/vig_dagger/toy_reach_env/sphere_environment.py','src/vig_dagger/toy_reach_env/environment.py',
                'src/vig_dagger/policy.py']:
        shutil.copy2(repo/rel,root/'source_snapshot'/Path(rel).name)
    setup=root.parent/'setup';env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',
        PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='2',
        VK_ICD_FILENAMES=str(setup/'nvidia-570/nvidia_icd_local.json'),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(setup/'nvidia-570/10_nvidia_local.json'))
    env['LD_LIBRARY_PATH']=str(setup/'nvidia-graphics')+':'+env.get('LD_LIBRARY_PATH','')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    processes=[]
    for gpu,name in enumerate(NAMES):
        status=root/name/'status.json'
        if status.exists() and json.loads(status.read_text())['state']=='complete':continue
        cmd=[str(repo/'.venv-isaac/bin/python'),str(repo/'scripts/rollout_sphere_simple.py'),
             '--root',str(root),'--name',name,'--gpu',str(gpu)]
        with (root/'logs'/f'{name}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,p));print(f'LAUNCHED {name} pid={p.pid} gpu={gpu}',flush=True)
    write(root/'processes.json',[dict(name=n,pid=p.pid) for n,p in processes])
    started=time.time()
    while True:
        statuses={}
        for name in NAMES:
            p=root/name/'status.json';statuses[name]=json.loads(p.read_text()) if p.exists() else {'state':'starting','completed':0}
        write(root/'pipeline.json',dict(state='running',pid=os.getpid(),elapsed_seconds=time.time()-started,
            completed=sum(s.get('completed',0) for s in statuses.values()),total=150,policies=statuses))
        for name,p in processes:
            if p.poll() is not None and (p.returncode or statuses[name]['state']!='complete'):
                raise RuntimeError(f'{name} exited with code {p.returncode}: {statuses[name]}')
        if all(s['state']=='complete' for s in statuses.values()):break
        time.sleep(10)
    report(root)
    write(root/'pipeline.json',dict(state='complete',completed=150,total=150,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','report']);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    try:globals()[args.mode](args.root)
    except BaseException as exc:
        write(args.root/'pipeline.json',dict(state='failed',error=repr(exc)));traceback.print_exc();raise
