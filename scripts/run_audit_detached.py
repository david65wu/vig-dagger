"""Launch CPU-only analysis independently of SSH; refuse a duplicate live renderer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work-dir',type=Path,required=True)
a=p.parse_args()
root=a.work_dir.resolve(); dest=root/'results/dagger_audit';dest.mkdir(parents=True,exist_ok=True)
pidfile=dest/'audit.pid'
if pidfile.exists():
    pid=int(pidfile.read_text())
    proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists() and b'vig_dagger.audit' in proc.read_bytes():
        raise SystemExit(f'Analysis already running: PID {pid}')
env=os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
with (dest/'audit.log').open('a') as log:
    proc=subprocess.Popen(['nice','-n','15',sys.executable,'-m','vig_dagger.audit','--work-dir',str(root),'--watch'],
                          cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(f'{proc.pid}\n')
print(f'Detached analysis PID {proc.pid}; start at {dest / "README.md"}')
