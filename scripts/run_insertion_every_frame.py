"""Detached dense/MoE insertion evaluation with one score per video frame."""
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
import tempfile
import threading
import time

from run_insertion_experiment import REPO,write_json,run_job,alive,gpu_env
from run_insertion_moe import checkpoint_ready

BASE=Path('/cpfs02/user/caojianuo/vig-dagger-runs')
EPOCHS=[10,25,50,100,200,300]


def prepare(root,variant):
    source=BASE/'charger_insertion_analysis_v1';root.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'cases.json',root/'cases.json')
    for item in json.loads((source/'cases.json').read_text()):
        dst=root/item['state_file'];dst.parent.mkdir(exist_ok=True)
        if not dst.exists():os.link(source/item['state_file'],dst)
    protocol=json.loads((source/'protocol.json').read_text())
    protocol.update(vig_enabled=variant=='moe',full_rollout_videos=True,score_stride=1,calibration_stride=1,
        policy_label=('7-level MoE' if variant=='moe' else 'Dense full-resolution')+' | scores every frame',
        calibration='Every recorded pre-action frame from all own training demos; separate references for flow and VIG',
        videos='All 50 episodes per checkpoint; one flow score per frame, and one VIG score per frame for MoE',
        score_alignment='Score and rendered image use the same pre-action observation',
        executed_actions_per_replan=8,vig_samples=32,loss_draws=512,
        vig_metric='Maximum mean per-cell normalized-action 1D Wasserstein distance across six degraded factors; 32 common-noise samples per level')
    write_json(root/'protocol.json',protocol)
    if not (root/'index.html').exists():
        links=''.join(f'<li><a href="runs/demos{n}_epoch{e:03d}/index.html">{n} demos / epoch {e}</a></li>' for n in [50,200,500] for e in EPOCHS)
        (root/'index.html').write_text('<html><meta charset="utf-8"><body><h1>'+protocol['policy_label']+'</h1><p>Evaluation in progress. 50 rollout videos per checkpoint; scores on every video frame. Galleries appear as episodes complete.</p><ul>'+links+'</ul></body></html>')


def preflight(pipeline):
    import numpy as np
    import torch
    from vig_dagger.insertion_analysis import run_rollout
    from vig_dagger.insertion_env import make_insertion_env
    from vig_dagger.policy import LibraryPolicy,PolicySet
    from vig_dagger.scoring import VIGScorer
    torch.set_num_threads(2)
    case=json.loads((BASE/'charger_insertion_analysis_v1/cases.json').read_text())[0]
    results=[]
    for variant in ['dense','moe']:
        training=BASE/('charger_insertion_v1' if variant=='dense' else 'charger_insertion_moe7_v1')
        checkpoint=training/'policies'/f'demos50_{"dense_full" if variant=="dense" else "moe_7level"}'/'epoch-300.pt'
        policy=LibraryPolicy(checkpoint,'cuda:0')
        old=BASE/('charger_insertion_analysis_v1' if variant=='dense' else 'charger_insertion_moe7_flow_vig_eval_v1')/'runs/demos50_epoch300'
        reference=np.sort(np.load(old/'calibration.npz')['flow'])
        context=None
        if variant=='moe':context=dict(policies=PolicySet({1:policy}),scorer=VIGScorer(32),reference=np.sort(np.load(old/'calibration_vig.npz')['vig']))
        env=make_insertion_env(render_backend='cuda:0');out=Path(tempfile.mkdtemp(prefix=f'insertion_frame_{variant}_smoke_'))
        try:row=run_rollout(policy,env,case,BASE/'charger_insertion_analysis_v1',out,reference,'CADENCE TEST',{},vig_context=context,full_video=True,score_stride=1,max_steps=16)
        finally:env.close()
        trace=np.load(out/row['trace'])
        assert row['steps']==16 and np.array_equal(trace['score_steps'],np.arange(16))
        assert np.array_equal(trace['replan_steps'],[0,8])
        for step,action in enumerate(trace['actions']):assert np.array_equal(action,trace['replan_chunks'][step//8,step%8])
        if variant=='moe':assert len(trace['vig'])==16 and context['rng_preservation_verified']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames','-of','json',str(out/row['video'])]))
        assert int(probe['streams'][0]['nb_frames'])==16
        results.append(dict(variant=variant,frames=16,scores=16,replans=2,executed_actions_verified=True,test_artifacts=str(out),
            calibration_note='Cadence smoke test uses prior references only; production recalibrates on every training frame'))
        del policy,context;torch.cuda.empty_cache()
    write_json(pipeline/'preflight.json',dict(passed=True,results=results,finished_at=time.time()))


def coordinate(pipeline):
    dense=BASE/'charger_insertion_dense_every_frame_v1';moe=BASE/'charger_insertion_moe7_every_frame_v1'
    training_moe=BASE/'charger_insertion_moe7_v1'
    roots={'dense':dense,'moe':moe}
    for variant,root in roots.items():prepare(root,variant)
    if not (pipeline/'preflight.json').exists():
        run_job([sys.executable,'-u','scripts/run_insertion_every_frame.py','--pipeline',str(pipeline),'--preflight'],pipeline/'preflight.log',0,pipeline/'preflight_process.json','--preflight')
    assert json.loads((pipeline/'preflight.json').read_text())['passed']
    jobs=[(variant,n,e) for variant in ['moe','dense'] for n in [500,200,50] for e in EPOCHS
        if not (roots[variant]/'runs'/f'demos{n}_epoch{e:03d}'/'complete.json').exists()]
    jobs.sort(key=lambda x:({('moe',500):0,('dense',500):1,('moe',200):2,('dense',200):3,('moe',50):4,('dense',50):5}[x[:2]],x[2]))
    lock=threading.Lock();stop=threading.Event()
    def status(phase,**kw):
        row=dict(phase=phase,pid=os.getpid(),updated_at=time.time(),analysis_roots={k:str(v) for k,v in roots.items()},**kw)
        write_json(pipeline/'pipeline.json',row)
        write_json(training_moe/'pipeline.json',row)
    status('calibration_and_evaluation',checkpoints=36,rollouts=1800,gpus=list(range(8)))
    for root in roots.values():write_json(root/'pipeline.json',dict(phase='evaluating',score_stride=1,calibration_stride=1,updated_at=time.time()))
    def lane(gpu):
        try:
            # The one remaining training worker owns GPU 6 until it finishes.
            for count in [50,200,500]:
                out=training_moe/'policies'/f'demos{count}_moe_7level';registry=out/'process.json'
                if registry.exists():
                    process=json.loads(registry.read_text())
                    if process['gpu']==gpu and alive(process['pid'],str(out)):
                        run_job(process['command'],training_moe/'logs'/f'train_demos{count}.log',gpu,registry,str(out))
                        assert json.loads((out/'status.json').read_text())['state']=='complete'
            while not stop.is_set():
                with lock:
                    if not jobs:return
                    job=next((x for x in jobs if x[0]=='dense' or checkpoint_ready(training_moe,x[1],x[2])),None)
                    if job is not None:jobs.remove(job)
                if job is None:stop.wait(10);continue
                variant,count,epoch=job;root=roots[variant];name=f'demos{count}_epoch{epoch:03d}';out=root/'runs'/name
                training=BASE/('charger_insertion_v1' if variant=='dense' else 'charger_insertion_moe7_v1')
                write_json(pipeline/f'lane{gpu}.json',dict(variant=variant,checkpoint=name,phase='evaluating',updated_at=time.time()))
                run_job([sys.executable,'-u','-m','vig_dagger.insertion_analysis','worker','--root',str(root),'--training',str(training),'--count',str(count),'--epoch',str(epoch)],root/'logs'/f'{name}.log',gpu,root/'processes'/f'{name}.json',f'--root {root} --training {training} --count {count} --epoch {epoch}')
                assert (out/'complete.json').exists()
        except BaseException as exc:
            stop.set();status('failed',gpu=gpu,error=repr(exc));raise
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,g) for g in range(8)]
        for future in futures:future.result()
    status('plotting')
    for variant,root in roots.items():
        assert len(list((root/'runs').glob('*/complete.json')))==18
        subprocess.run([sys.executable,'scripts/plot_insertion_analysis.py','--root',str(root)],cwd=REPO,check=True)
        write_json(root/'pipeline.json',dict(phase='complete',checkpoints=18,rollouts=900,score_stride=1,calibration_stride=1,finished_at=time.time()))
    subprocess.run([sys.executable,'scripts/compare_insertion_variants.py','--dense',str(dense),'--moe',str(moe)],cwd=REPO,check=True)
    write_json(BASE/'charger_insertion_analysis_v1/analysis_selection.json',dict(analysis_root=str(dense),complete=True,reason='Every-frame rollout evaluation',updated_at=time.time()))
    status('complete',checkpoints=36,rollouts=1800)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();a.pipeline.mkdir(parents=True,exist_ok=True)
    if a.preflight:preflight(a.pipeline)
    else:
        with (BASE/'charger_insertion_moe7_v1/coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:coordinate(a.pipeline)
            except BaseException as exc:
                write_json(a.pipeline/'pipeline.json',dict(phase='failed',error=repr(exc),updated_at=time.time()));raise
