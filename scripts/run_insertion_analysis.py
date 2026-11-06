"""Detached coordinator for all 18 calibrated evaluations and final figures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import queue
import sys
import time
import traceback
import subprocess

from run_insertion_experiment import run_job,alive,write_json,REPO


def run(root,training):
    def status(phase,**kwargs):write_json(root/'pipeline.json',dict(phase=phase,updated_at=time.time(),**kwargs))
    status('preparing_states')
    process=json.loads((root/'prepare_process.json').read_text())
    while alive(process['pid'],str(root)):time.sleep(5)
    if not (root/'cases.json').exists():raise RuntimeError('State preparation failed; see prepare.log')
    status('pilot')
    pilot=root/'pilot/demos50_epoch300/complete.json'
    if not pilot.exists():
        run_job([sys.executable,'-u','-m','vig_dagger.insertion_analysis','pilot','--root',str(root),'--training',str(training)],
            root/'pilot.log',6,root/'pilot_process.json',str(root))
    if not pilot.exists():raise RuntimeError('Pilot incomplete')
    # An explicit local readiness marker is written after visual inspection of
    # the pilot, before launching hundreds of rollouts.
    status('awaiting_pilot_review')
    while not (root/'pilot_review.json').exists():time.sleep(5)
    jobs=queue.Queue()
    for count in [50,200,500]:
        for epoch in [10,25,50,100,200,300]:jobs.put((count,epoch))
    status('evaluating',total_checkpoints=18,gpus=[6,7])
    def lane(gpu):
        while True:
            try:count,epoch=jobs.get_nowait()
            except queue.Empty:return
            name=f'demos{count}_epoch{epoch:03d}';out=root/'runs'/name
            if not (out/'complete.json').exists():
                run_job([sys.executable,'-u','-m','vig_dagger.insertion_analysis','worker','--root',str(root),
                    '--training',str(training),'--count',str(count),'--epoch',str(epoch)],
                    root/'logs'/f'{name}.log',gpu,root/'processes'/f'{name}.json',f'--count {count} --epoch {epoch}')
            if not (out/'complete.json').exists():raise RuntimeError(f'Incomplete evaluation {name}')
            completed=len(list((root/'runs').glob('*/complete.json')))
            # Distinct files avoid simultaneous atomic-write collisions.
            write_json(root/f'lane{gpu}.json',dict(completed_checkpoints=completed,last_finished=name,updated_at=time.time()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,gpu) for gpu in [6,7]]
        for future in futures:future.result()
    status('plotting')
    subprocess.run([sys.executable,'scripts/plot_insertion_analysis.py','--root',str(root)],cwd=REPO,check=True)
    status('complete',checkpoints=18,rollouts=900)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args.root,args.training)
        except BaseException as exc:
            write_json(args.root/'pipeline.json',dict(phase='failed',error=repr(exc),updated_at=time.time()))
            traceback.print_exc();raise
