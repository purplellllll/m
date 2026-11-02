"""Finite serial CPU queue; stop on failure, never auto-retry failed experiments."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(data):
    path=ROOT/'pipeline_status.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def main():
    started=time.time();done=[]
    for version in ['v3_multi','v4_epoch6']:
        for phase,script in [('search','run_experiment.py'),('evaluation','evaluate_version.py')]:
            receipt=ROOT/version/('search_complete.json' if phase=='search' else 'evaluation_complete.json')
            if receipt.exists():
                data=json.loads(receipt.read_text(encoding='utf-8'));assert data['status']=='completed'
                done.append(f'{version}/{phase}');continue
            save(dict(stage='running',version=version,phase=phase,completed=done,started_unix=started,updated_unix=time.time()))
            with (ROOT/f'{version}_{phase}.log').open('a',encoding='utf-8') as handle:
                subprocess.run([sys.executable,'-u',str(ROOT/script),version],stdout=handle,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            done.append(f'{version}/{phase}')
    save(dict(stage='rendering_and_auditing',completed=done,started_unix=started,updated_unix=time.time()))
    subprocess.run([sys.executable,str(ROOT/'summarize_and_verify.py')],check=True,cwd=ROOT)
    save(dict(stage='completed',completed=done,started_unix=started,completed_unix=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(dict(stage='failed',error=repr(error),unix=time.time(),auto_retry=False));raise
