"""Detached coordinator: validate sphere collections, launch six trainings, track completion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil
from vig_dagger.toy_training import write_json


def run(root):
    repo=Path(__file__).resolve().parents[1]
    jobs=json.loads((root/'collection_processes.json').read_text())
    started=time.time()
    def status(state,**extra):
        write_json(root/'pipeline.json',dict(state=state,pid=os.getpid(),updated_at=time.time(),**extra))
    status('collecting')
    while True:
        missing=[]
        for name,job in jobs.items():
            if (Path(job['out'])/'summary.json').exists():continue
            missing.append(name)
            try:
                process=psutil.Process(job['pid'])
                alive=process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
            except psutil.Error:
                alive=False
            if not alive:raise RuntimeError(f'Collection stopped before completion: {name}')
        if not missing:break
        if time.time()-started>3600:raise TimeoutError('Collection exceeded one hour')
        time.sleep(5)
    status('validating')
    def validate(job):
        path=Path(job['out'])
        if (path/'validation.json').exists():return
        subprocess.run([sys.executable,str(repo/'scripts/validate_toy_reach_dataset.py'),str(path)],cwd=repo,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(validate,jobs.values()))
    training=next(Path(j['out']) for j in jobs.values() if j['num_demos']==20)
    validation=next(Path(j['out']) for j in jobs.values() if j['num_demos']==5)
    train_seeds={r['seed'] for r in json.loads((training/'summary.json').read_text())['demos']}
    val_seeds={r['seed'] for r in json.loads((validation/'summary.json').read_text())['demos']}
    assert not train_seeds & val_seeds
    status('preparing')
    if not (root/'experiment.json').exists():
        subprocess.run([sys.executable,'-m','vig_dagger.toy_training','prepare','--task','sphere',
            '--root',str(root),'--source',str(training/'dataset'),'--validation',str(validation/'dataset')],cwd=repo,check=True)
    for relative in ['scripts/collect_toy_reach.py','scripts/validate_toy_reach_dataset.py',
                     'scripts/start_sphere_training.py','src/vig_dagger/toy_training.py',
                     'src/vig_dagger/toy_reach_env/sphere_config.py','src/vig_dagger/toy_reach_env/sphere_environment.py',
                     'src/vig_dagger/toy_reach_env/sphere_sampling.py','src/vig_dagger/toy_reach_env/planner.py']:
        path=root/'source_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/relative,path)
    subprocess.run([sys.executable,'-m','vig_dagger.toy_training','launch','--root',str(root)],cwd=repo,check=True)
    specs=json.loads((root/'experiment.json').read_text())['runs']
    while True:
        states={}
        for spec in specs:
            p=Path(spec['run_dir'])/'status.json'
            states[spec['name']]=json.loads(p.read_text()) if p.exists() else {'state':'starting'}
        failed=[name for name,s in states.items() if s['state']=='failed']
        if failed:raise RuntimeError(f'Training workers failed: {failed}')
        complete=sum(s['state']=='complete' for s in states.values())
        status('training',completed=complete,total=6)
        if complete==6:break
        time.sleep(15)
    summary=[]
    for spec in specs:
        out=Path(spec['run_dir'])
        assert (out/'latest.pt').is_file() and (out/'epoch-100.pt').is_file()
        metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        training_losses=[m['train/loss'] for m in metrics if 'train/loss' in m]
        validation_losses=[m['val/loss'] for m in metrics if 'val/loss' in m]
        assert training_losses and validation_losses and all(math.isfinite(x) for x in training_losses+validation_losses)
        summary.append(dict(name=spec['name'],epochs=100,updates=spec['total_updates'],
            final_train_loss=training_losses[-1],final_validation_loss=validation_losses[-1],
            checkpoint=str(out/'latest.pt')))
    write_json(root/'training_summary.json',summary)
    status('complete',completed=6,total=6)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    try:
        run(args.root.resolve())
    except BaseException as exc:
        write_json(args.root/'pipeline.json',dict(state='failed',error=repr(exc),pid=os.getpid()))
        traceback.print_exc();raise
