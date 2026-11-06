"""Detached, restartable insertion collection and training coordinator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
PREVIEW = Path('/cpfs02/user/caojianuo/vig-dagger-runs/charger_insertion_preview_v1')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def gpu_env(gpu):
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
        HF_HOME='/cpfs02/user/caojianuo/.cache/huggingface', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true',
        VK_ICD_FILENAMES='/cpfs02/user/caojianuo/vig-dagger-runs/setup/nvidia-570/nvidia_icd_local.json',
        __EGL_VENDOR_LIBRARY_FILENAMES='/cpfs02/user/caojianuo/vig-dagger-runs/setup/nvidia-570/10_nvidia_local.json')
    env['LD_LIBRARY_PATH']='/cpfs02/user/caojianuo/vig-dagger-runs/setup/nvidia-graphics:'+env.get('LD_LIBRARY_PATH','')
    return env


def alive(pid, marker):
    try:
        import psutil
        p = psutil.Process(pid)
        return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and marker in ' '.join(p.cmdline())
    except (OSError, psutil.Error):
        return False


def run_job(command, log_path, gpu, registry, marker):
    if registry.exists():
        previous=json.loads(registry.read_text())
        if alive(previous['pid'],marker):
            print(f'Adopting PID {previous["pid"]}: {marker}',flush=True)
            while alive(previous['pid'],marker): time.sleep(5)
            return
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a') as log:
        p=subprocess.Popen(command,cwd=REPO,env=gpu_env(gpu),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(registry,dict(pid=p.pid,gpu=gpu,command=command,started_at=time.time()))
    print(f'Started GPU {gpu}, PID {p.pid}: {marker}',flush=True)
    result=p.wait()
    if result: raise RuntimeError(f'Job exited {result}: {marker}; log: {log_path}')


def collect_part(root, index, gpu):
    part=root/'collections'/f'part_{index:02d}'
    marker=part/'complete.json'
    if marker.exists(): return json.loads(marker.read_text())
    part.mkdir(parents=True,exist_ok=True)
    # A completed attempt is reusable even if the coordinator stopped before
    # publishing its pointer. Incomplete raw recordings are preserved.
    for previous in sorted(part.glob('attempt_*')):
        if (previous/'process.json').exists():
            process=json.loads((previous/'process.json').read_text())
            while alive(process['pid'],str(previous)): time.sleep(5)
        if (previous/'complete.json').exists():
            result=dict(out=str(previous),dataset=str(previous/'dataset'),part=index)
            write_json(marker,result);return result
    number=len(list(part.glob('attempt_*')))
    out=part/f'attempt_{number:02d}';out.mkdir()
    count=45 if index==9 else 50
    command=[sys.executable,'scripts/collect_insertion_preview.py','--bulk','--out',str(out),
        '--num-demos',str(count),'--start-seed',str(1000+index*10000),
        '--max-attempts',str(count*10),'--gpu',str(gpu),'--trajectory-name',f'part_{index:02d}']
    run_job(command,out/'collection.log',gpu,out/'process.json',str(out))
    if not (out/'complete.json').exists(): raise RuntimeError(f'Incomplete collection {out}')
    result=dict(out=str(out),dataset=str(out/'dataset'),part=index)
    write_json(marker,result);return result


def coordinate(root):
    def status(state,**kwargs):
        write_json(root/'pipeline.json',dict(state=state,pid=os.getpid(),updated_at=time.time(),**kwargs))
    status('collecting',target_demos=500,retained_preview_demos=5,gpus=[6,7])
    def lane(gpu, indices):
        return [collect_part(root,i,gpu) for i in indices]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,6,range(0,10,2)),pool.submit(lane,7,range(1,10,2))]
        parts=[p for future in futures for p in future.result()]
    parts.sort(key=lambda p:p['part'])
    write_json(root/'collection_sources.json',dict(preview=str(PREVIEW),parts=parts))
    status('preparing_training',collected_demos=500,gpus=[6,7])
    subprocess.run([sys.executable,'-m','vig_dagger.insertion_training','prepare','--root',str(root)],cwd=REPO,check=True)
    if not (root/'training_preflight.json').exists():
        run_job([sys.executable,'-m','vig_dagger.insertion_training','preflight','--root',str(root)],
            root/'logs/preflight.log',6,root/'preflight_process.json',str(root))
    if not (root/'training_preflight.json').exists(): raise RuntimeError('Training preflight did not pass')
    status('training',collected_demos=500,gpus=[6,7],schedule={'6':[500],'7':[200,50]})
    def train_lane(gpu, counts):
        for count in counts:
            out=root/'policies'/f'demos{count}_dense_full'
            if (out/'latest.pt').exists(): continue
            run_job([sys.executable,'-m','vig_dagger.insertion_training','train','--run-dir',str(out),'--gpu',str(gpu)],
                root/'logs'/f'train_demos{count}.log',gpu,out/'process.json',str(out))
            if not (out/'latest.pt').exists(): raise RuntimeError(f'Training did not finish: {out}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train_lane,6,[500]),pool.submit(train_lane,7,[200,50])]
        for future in futures: future.result()
    status('training_complete_analysis_pending',collected_demos=500,
        note='All six epoch checkpoints retained per dataset size; rollout calibration and analysis are next.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: coordinate(root)
        except BaseException as exc:
            write_json(root/'pipeline.json',dict(state='failed',pid=os.getpid(),updated_at=time.time(),error=repr(exc)))
            traceback.print_exc();raise
