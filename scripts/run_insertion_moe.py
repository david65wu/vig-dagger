"""Run matched MoE insertion training, calibration, rollouts, plots and videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

from run_insertion_experiment import run_job,write_json,REPO,alive

FACTORS=[1,2,4,8,16,32,64]
EPOCHS=[10,25,50,100,200,300]


def prepare(root,dense,analysis,dense_analysis,flow_vig=False,prior_analysis=None):
    from robopolicy import RunConfig,build_dataset
    root.mkdir(parents=True,exist_ok=True);analysis.mkdir(parents=True,exist_ok=True)
    if not (root/'experiment.json').exists():
        for name in ['demo_order.json','collection_sources.json']:
            shutil.copy2(dense/name,root/name)
        if not (root/'datasets').exists():
            (root/'datasets').symlink_to(dense/'datasets',target_is_directory=True)
        template=json.loads((REPO/'src/vig_dagger/configs/policy.json').read_text())
        dense_experiment=json.loads((dense/'experiment.json').read_text());runs=[]
        for old in dense_experiment['runs']:
            old_dir=Path(old['run_dir']);n=old['num_demos'];out=root/'policies'/f'demos{n}_moe_7level'
            out.mkdir(parents=True,exist_ok=True)
            original=json.loads((old_dir/'run_config.json').read_text());config=copy.deepcopy(original)
            config['policy']['head']['moe']=template['policy']['head']['moe']
            config['policy']['conditioning']['vocab_sizes']=[7]
            config['processing']['resolution']['factors']=FACTORS
            config['logging']['out_dir']=str(out)
            validated=RunConfig.model_validate(config)
            dataset=build_dataset(validated.dataset,validated.features,'train')
            assert dataset.fingerprint==old['dataset_fingerprint']
            assert [list(x) for x in dataset.selection]==old['selection']
            assert len(dataset)==old['training_samples']
            dataset.close()
            # Architecture/resolution/output path are the only config changes.
            comparable=copy.deepcopy(config)
            comparable['policy']['head']['moe']=original['policy']['head']['moe']
            comparable['policy']['conditioning']=original['policy']['conditioning']
            comparable['processing']['resolution']=original['processing']['resolution']
            comparable['logging']['out_dir']=original['logging']['out_dir']
            assert comparable==original
            spec=dict(old,name=out.name,run_dir=str(out),variant='moe_7level',
                resolution_factors=FACTORS,normalization_reference=str(old_dir/'normalization.json'),
                dense_reference=str(old_dir))
            write_json(out/'run_config.json',validated.model_dump(mode='json'))
            write_json(out/'experiment.json',spec);runs.append(spec)
        write_json(root/'experiment.json',dict(task='charger_insertion',variant='moe_7level',
            cameras=['hand_camera','side_camera'],runs=runs,checkpoint_epochs=EPOCHS,
            resolution_factors=FACTORS,evaluation_factor=1,source_dense_experiment=str(dense),
            matched_datasets=True,matched_training_hyperparameters=True,created_at=time.time()))
    cases=json.loads((dense_analysis/'cases.json').read_text())
    for row in cases:
        src=dense_analysis/row['state_file'];dst=analysis/row['state_file'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():os.link(src,dst)
        assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest()
    for name in ['cases.json','rejected_setups.json','environment_validation.json']:
        shutil.copy2(dense_analysis/name,analysis/name)
    protocol=json.loads((dense_analysis/'protocol.json').read_text())
    protocol.update(training_root=str(root),variant='moe_7level',policy_label='7-level MoE (full-resolution evaluation)',
        dense_reference=str(dense_analysis),resolution_factors=FACTORS,evaluation_factor=1,
        physical_states='Byte-identical snapshots from the dense-policy evaluation',video_y_ticks=[0,20,40,60,80,100])
    if flow_vig:
        protocol.update(vig_enabled=True,full_rollout_videos=True,vig_samples=32,vig_factors=FACTORS,
            vig_metric='Maximum over degraded levels of mean per-cell 1D Wasserstein distance on 16x8 normalized actions; shared noise across levels',
            vig_calibration='Own checkpoint training observations, same stride-8 frames as flow; independent empirical percentile reference',
            videos='All 50 rollouts per checkpoint; flow percentile above VIG percentile, raw values, and all 16 action targets',
            diagnostic_rng='VIG runs in a forked RNG context; full-resolution flow chunks still drive the robot',
            rollout_generation='All 900 episodes freshly evaluated with simultaneous flow and VIG; previous flow-only evaluation retained separately',
            previous_flow_analysis=str(prior_analysis) if prior_analysis else None)
        if prior_analysis:
            for count in [50,200,500]:
                for epoch in EPOCHS:
                    name=f'demos{count}_epoch{epoch:03d}';source=prior_analysis/'runs'/name;dest=analysis/'runs'/name
                    if (source/'calibration.json').exists() and not (dest/'calibration.npz').exists():
                        meta=json.loads((source/'calibration.json').read_text())
                        checkpoint=root/'policies'/f'demos{count}_moe_7level'/f'epoch-{epoch:03d}.pt'
                        assert meta['sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                        dest.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(source/'calibration.npz',dest/'calibration.npz')
                        write_json(dest/'flow_calibration_reuse.json',dict(source=str(source),checkpoint_sha256=meta['sha256']))
    write_json(analysis/'protocol.json',protocol)
    write_json(analysis/'matched_dense_cases.json',dict(passed=True,cases=len(cases),dense_reference=str(dense_analysis),
        state_sha256={r['state_file']:hashlib.sha256((analysis/r['state_file']).read_bytes()).hexdigest() for r in cases}))
    if flow_vig and not (analysis/'index.html').exists():
        links=''.join(f'<li><a href="runs/demos{n}_epoch{e:03d}/index.html">{n} demos / epoch {e}</a></li>' for n in [50,200,500] for e in EPOCHS)
        (analysis/'index.html').write_text('<html><meta charset="utf-8"><body><h1>MoE flow + VIG evaluation</h1><p>In progress. Checkpoint galleries appear as rollouts complete. Each will contain 50 videos with flow and VIG plots. Aggregate figures are generated after all 18 checkpoints finish.</p><ul>'+links+'</ul></body></html>')


def coordinate(root,dense,analysis,dense_analysis):
    def status(phase,**kwargs):write_json(root/'pipeline.json',dict(phase=phase,pid=os.getpid(),updated_at=time.time(),analysis_root=str(analysis),**kwargs))
    status('preparing')
    prepare(root,dense,analysis,dense_analysis)
    status('preflight')
    if not (root/'training_preflight.json').exists():
        run_job([sys.executable,'-u','-m','vig_dagger.insertion_training','preflight','--root',str(root),'--variant','moe_7level'],
            root/'logs/preflight.log',6,root/'preflight_process.json',str(root))
    assert json.loads((root/'training_preflight.json').read_text())['variant']=='moe_7level'
    status('training',gpus=[6,7],schedule={'6':[500],'7':[200,50]})
    def train_lane(gpu,counts):
        for count in counts:
            out=root/'policies'/f'demos{count}_moe_7level'
            if not (out/'latest.pt').exists():
                run_job([sys.executable,'-u','-m','vig_dagger.insertion_training','train','--run-dir',str(out),'--gpu',str(gpu)],
                    root/'logs'/f'train_demos{count}.log',gpu,out/'process.json',str(out))
            assert (out/'latest.pt').exists()
            for epoch in EPOCHS:assert (out/f'epoch-{epoch:03d}.pt').exists()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train_lane,6,[500]),pool.submit(train_lane,7,[200,50])]
        for future in futures:future.result()
    status('calibration_and_evaluation',checkpoints=18,rollouts=900)
    write_json(analysis/'pipeline.json',dict(phase='evaluating',updated_at=time.time(),total_checkpoints=18,gpus=[6,7]))
    jobs=queue.Queue()
    for count in [50,200,500]:
        for epoch in EPOCHS:jobs.put((count,epoch))
    def evaluate_lane(gpu):
        while True:
            try:count,epoch=jobs.get_nowait()
            except queue.Empty:return
            name=f'demos{count}_epoch{epoch:03d}';out=analysis/'runs'/name
            if not (out/'complete.json').exists():
                run_job([sys.executable,'-u','-m','vig_dagger.insertion_analysis','worker','--root',str(analysis),
                    '--training',str(root),'--count',str(count),'--epoch',str(epoch)],
                    analysis/'logs'/f'{name}.log',gpu,analysis/'processes'/f'{name}.json',f'--count {count} --epoch {epoch}')
            assert (out/'complete.json').exists()
            write_json(analysis/f'lane{gpu}.json',dict(last_finished=name,updated_at=time.time(),
                completed_checkpoints=len(list((analysis/'runs').glob('*/complete.json')))))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(evaluate_lane,gpu) for gpu in [6,7]]
        for future in futures:future.result()
    status('plotting')
    subprocess.run([sys.executable,'scripts/plot_insertion_analysis.py','--root',str(analysis)],cwd=REPO,check=True)
    subprocess.run([sys.executable,'scripts/compare_insertion_variants.py','--dense',str(dense_analysis),'--moe',str(analysis)],cwd=REPO,check=True)
    write_json(analysis/'pipeline.json',dict(phase='complete',checkpoints=18,rollouts=900,updated_at=time.time()))
    status('complete',training_runs=3,checkpoints=18,rollouts=900)


def checkpoint_ready(root,count,epoch):
    out=root/'policies'/f'demos{count}_moe_7level'
    report=out/'status.json'
    if not report.exists():return False
    progress=json.loads(report.read_text())
    spec=json.loads((out/'experiment.json').read_text())
    required=next(int(step) for step,name in spec['checkpoints'].items()
        if name==f'epoch-{epoch:03d}.pt')
    # Reports are written after save() returns. File existence alone could
    # expose a checkpoint still being written by a live training worker.
    return (out/f'epoch-{epoch:03d}.pt').exists() and progress.get('step',0)>=required


def coordinate_all_gpus(root,dense,analysis,dense_analysis,flow_vig=False,prior_analysis=None):
    prepare(root,dense,analysis,dense_analysis,flow_vig,prior_analysis)
    if flow_vig:assert json.loads((analysis/'flow_vig_preflight.json').read_text())['passed']
    preflight=json.loads((root/'training_preflight.json').read_text())
    assert preflight['passed'] and preflight['variant']=='moe_7level'
    assignments={0:50,6:500,7:200}
    pending=[(count,epoch) for epoch in EPOCHS for count in [50,200,500]
        if not (analysis/'runs'/f'demos{count}_epoch{epoch:03d}'/'complete.json').exists()]
    lock=threading.Lock();stopped=threading.Event()
    def status(phase,**kwargs):
        write_json(root/'pipeline.json',dict(phase=phase,pid=os.getpid(),updated_at=time.time(),
            analysis_root=str(analysis),gpus=list(range(8)),training_gpus=assignments,**kwargs))
    status('training_and_evaluation',checkpoints=18,rollouts=900)
    write_json(analysis/'pipeline.json',dict(phase='evaluating_as_checkpoints_arrive',
        updated_at=time.time(),total_checkpoints=18,gpus=list(range(8))))
    def lane(gpu):
        try:
            if flow_vig and (root/'prior_analysis_workers.json').exists():
                for old in json.loads((root/'prior_analysis_workers.json').read_text()):
                    if old['gpu']==gpu:
                        while alive(old['pid'],'vig_dagger.insertion_analysis') and not stopped.is_set():stopped.wait(5)
            if gpu in assignments:
                count=assignments[gpu];out=root/'policies'/f'demos{count}_moe_7level'
                report=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                if report.get('state')!='complete':
                    # run_job adopts registered live workers without restarting them.
                    write_json(analysis/f'lane{gpu}.json',dict(phase='training',demos=count,updated_at=time.time()))
                    run_job([sys.executable,'-u','-m','vig_dagger.insertion_training','train','--run-dir',str(out),'--gpu',str(gpu)],
                        root/'logs'/f'train_demos{count}.log',gpu,out/'process.json',str(out))
                assert json.loads((out/'status.json').read_text())['state']=='complete'
                assert (out/'latest.pt').exists()
                for epoch in EPOCHS:assert checkpoint_ready(root,count,epoch)
            while not stopped.is_set():
                with lock:
                    if not pending:return
                    job=next((job for job in pending if checkpoint_ready(root,*job)),None)
                    if job is not None:pending.remove(job)
                if job is None:
                    write_json(analysis/f'lane{gpu}.json',dict(phase='waiting_for_checkpoint',updated_at=time.time()))
                    stopped.wait(10)
                    continue
                count,epoch=job;name=f'demos{count}_epoch{epoch:03d}';out=analysis/'runs'/name
                write_json(analysis/f'lane{gpu}.json',dict(phase='evaluating',checkpoint=name,updated_at=time.time()))
                run_job([sys.executable,'-u','-m','vig_dagger.insertion_analysis','worker','--root',str(analysis),
                    '--training',str(root),'--count',str(count),'--epoch',str(epoch)],
                    analysis/'logs'/f'{name}.log',gpu,analysis/'processes'/f'{name}.json',f'--count {count} --epoch {epoch}')
                assert (out/'complete.json').exists()
                write_json(analysis/f'lane{gpu}.json',dict(phase='available',last_finished=name,updated_at=time.time(),
                    completed_checkpoints=len(list((analysis/'runs').glob('*/complete.json')))))
        except BaseException as exc:
            stopped.set()
            with lock:status('failed',gpu=gpu,error=repr(exc))
            raise
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,gpu) for gpu in range(8)]
        for future in futures:future.result()
    assert len(list((analysis/'runs').glob('*/complete.json')))==18
    if flow_vig:
        for count in [50,200,500]:
            for epoch in EPOCHS:
                out=analysis/'runs'/f'demos{count}_epoch{epoch:03d}'
                rows=json.loads((out/'rollouts.json').read_text())
                assert len(rows)==50 and len({r['case_id'] for r in rows})==50
                assert all('video' in r and (out/r['video']).is_file() for r in rows)
                assert (out/'calibration_vig.json').is_file()
    status('plotting')
    subprocess.run([sys.executable,'scripts/plot_insertion_analysis.py','--root',str(analysis)],cwd=REPO,check=True)
    subprocess.run([sys.executable,'scripts/compare_insertion_variants.py','--dense',str(dense_analysis),'--moe',str(analysis)],cwd=REPO,check=True)
    write_json(analysis/'pipeline.json',dict(phase='complete',checkpoints=18,rollouts=900,updated_at=time.time()))
    status('complete',training_runs=3,checkpoints=18,rollouts=900)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--dense-analysis',type=Path,required=True)
    p.add_argument('--all-gpus',action='store_true',help='Adopt training and evaluate available checkpoints across GPUs 0–7')
    p.add_argument('--flow-vig',action='store_true',help='Calibrate and record both scores, with all rollout videos')
    p.add_argument('--prior-analysis',type=Path,help='Reuse flow reference arrays after verifying checkpoint hashes')
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.all_gpus:coordinate_all_gpus(args.root,args.dense,args.analysis,args.dense_analysis,args.flow_vig,args.prior_analysis)
            else:
                assert not args.flow_vig,'Flow/VIG evaluation uses the all-GPU scheduler'
                coordinate(args.root,args.dense,args.analysis,args.dense_analysis)
        except BaseException as exc:
            write_json(args.root/'pipeline.json',dict(phase='failed',error=repr(exc),updated_at=time.time()))
            traceback.print_exc();raise
