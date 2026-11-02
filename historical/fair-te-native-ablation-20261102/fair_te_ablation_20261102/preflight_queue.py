from pathlib import Path
import subprocess,json,time
root=Path(__file__).resolve().parent
work=root.parent.parent
python=work/'.runtime/scorer_cuda/Scripts/python.exe'
while not (work/'runs/te_stability_20261102/te_100000/training_complete.json').exists():
    time.sleep(10)
for arm in ['full','no_physics','no_graph','no_context','seq_transformer','seq2ribo_TE_adapt','RiboGL_TE_adapt','RiboDecode_TE_adapt']:
    with (root/('preflight_'+arm+'.log')).open('wb') as log:
        result=subprocess.run([str(python),'-u',str(root/'experiment.py'),'preflight','--arm',arm],cwd=work,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        (root/'preflight_failure.json').write_text(json.dumps(dict(arm=arm,returncode=result.returncode)),encoding='utf-8');raise SystemExit(result.returncode)
(root/'all_preflight_complete.json').write_text(json.dumps(dict(status='passed',arms=8)),encoding='utf-8')
