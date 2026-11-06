"""Detached per-policy pipeline: simulator rollouts, then offline probe scores."""
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
    out=args.root/args.run_dir.name; out.mkdir(parents=True,exist_ok=True)
    status=out/'pipeline.json'
    try:
        status.write_text(json.dumps({'state':'rollouts','pid':os.getpid(),'gpu':args.gpu}))
        subprocess.run([str(repo/'.venv-isaac/bin/python'),str(repo/'scripts/evaluate_toy_reach.py'),
                        '--mode','rollouts','--gpu',str(args.gpu),'--out',str(out),'--run-dir',str(args.run_dir)],check=True)
        status.write_text(json.dumps({'state':'waiting_for_probes','pid':os.getpid(),'gpu':args.gpu}))
        start=time.time()
        while not (args.root/'probes_done.json').exists():
            if time.time()-start>1800: raise TimeoutError('Shared probe capture did not complete in 30 minutes')
            time.sleep(5)
        status.write_text(json.dumps({'state':'scoring','pid':os.getpid(),'gpu':args.gpu}))
        subprocess.run([str(repo/'.venv/bin/python'),str(repo/'scripts/analyze_toy_reach.py'),'score',
                        '--root',str(args.root),'--out',str(out),'--run-dir',str(args.run_dir)],check=True)
        status.write_text(json.dumps({'state':'complete','pid':os.getpid(),'gpu':args.gpu}))
    except BaseException as exc:
        status.write_text(json.dumps({'state':'failed','error':repr(exc),'gpu':args.gpu}))
        traceback.print_exc()
        raise
