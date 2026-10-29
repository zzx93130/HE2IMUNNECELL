"""Wait for the verification slide, then run the remaining frozen evaluation."""
from pathlib import Path
import datetime, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/tcga_tls_golden_20261102'
LOCK=ROOT/'outputs/_gpu.lock'
FIRST=OUT/'slides/TCGA-BP-5192-01Z-00-DX1'
def status(state, **kw):
    p=OUT/'queue_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));tmp.replace(p)
def main():
    while LOCK.exists():
        if not (LOCK/'owner').exists():time.sleep(5);continue
        owner=(LOCK/'owner').read_text().strip()
        if not owner.startswith('tcga-golden pid '):
            status('blocked_by_another_gpu_job',owner=owner);return
        try:os.kill(int(owner.split()[-1]),0)
        except ProcessLookupError:
            status('stale_lock_requires_inspection',owner=owner);return
        status('verification_slide_running',owner=owner);time.sleep(30)
    if not (FIRST/'summary.json').exists() or (FIRST/'error.json').exists():
        status('verification_failed_inspect_slide_logs');return
    with open(OUT/'run.log','a') as log:
        proc=subprocess.Popen([sys.executable,str(ROOT/'deployment/tcga_tls_golden/evaluate.py'),'run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            status('running',worker_pid=proc.pid);time.sleep(30)
        status('finished' if proc.returncode==0 else 'failed_inspect_run_log',exit_code=proc.returncode)
if __name__=='__main__':main()
