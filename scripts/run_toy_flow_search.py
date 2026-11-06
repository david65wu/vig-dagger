"""Detached and restartable per-policy flow-search pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=args.root/args.run_dir.name;out.mkdir(parents=True,exist_ok=True)
    def status(state,**extra):
        (out/'pipeline.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),gpu=args.gpu,**extra)))
    try:
        status('waiting_for_capture')
        started=time.time()
        while not (args.root/'capture_done.json').exists():
            if time.time()-started>1800:raise TimeoutError('Probe capture did not finish in 30 minutes')
            time.sleep(5)
        if not (out/'scores_done.json').exists():
            status('scoring')
            subprocess.run([str(repo/'.venv/bin/python'),str(repo/'scripts/analyze_toy_flow_search.py'),
                'score','--root',str(args.root),'--run-dir',str(args.run_dir)],check=True)
        if not (out/'rollouts_done.json').exists():
            status('waiting_for_restore_validation')
            started=time.time()
            while not (args.root/'restore_validation.json').exists():
                if time.time()-started>900:raise TimeoutError('Snapshot restore validation did not finish in 15 minutes')
                time.sleep(5)
            status('rollouts')
            subprocess.run([str(repo/'.venv-isaac/bin/python'),str(repo/'scripts/toy_flow_search_sim.py'),
                'rollouts','--root',str(args.root),'--run-dir',str(args.run_dir),'--gpu',str(args.gpu)],check=True)
        status('complete')
    except BaseException as exc:
        status('failed',error=repr(exc));traceback.print_exc();raise
