"""Run the actual shipped RiboDecode algorithm at a declared finite budget."""
from pathlib import Path
import json,hashlib,subprocess,sys,time
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;OUT=HERE/'native_compare'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def linux(p):
    p=Path(p).resolve();return '/mnt/'+p.drive[0].lower()+'/'+p.relative_to(p.anchor).as_posix()
def prepare():
    OUT.mkdir(exist_ok=True)
    source=WORK/'tools/native_default_cpu.py';text=source.read_text(encoding='utf-8')
    before="choices=('parity','preflight','full')";assert text.count(before)==1
    text=text.replace(before,"choices=('parity','preflight','full','bounded')")
    before="package=Path(__file__).resolve().parents[1]/'external/ribodecode_1_3_0/RiboDecode'";assert text.count(before)==1
    text=text.replace(before,'package=Path('+repr(linux(WORK/'external/ribodecode_1_3_0/RiboDecode'))+')')
    before="    if changes:\n        config['training'].update(changes)";assert text.count(before)==1
    text=text.replace(before,"    if args.mode=='bounded':changes=dict(optim_num=800,optim_batchsize=20,num_workers=2,num_seqs_train_gen=2048,num_seqs_select_mfe=512,num_seqs_train_mfe=8192,num_optim_top=80)\n"+before)
    text=text.replace("{'parity':1,'preflight':2,'full':20}[args.mode]","{'parity':1,'preflight':2,'full':20,'bounded':3}[args.mode]")
    wrapper=OUT/'native_bounded_cpu.py'
    if wrapper.exists():assert wrapper.read_text(encoding='utf-8')==text
    else:wrapper.write_text(text,encoding='utf-8')
    inputs=read(HERE/'lambda_scan/inputs_locked.json')
    protocol=dict(task='Native RiboDecode1.3.0 versus continued human TE guided V3',cases=[dict(uid=r['uid'],gene=r['gene'],n_codons=r['n_codons']) for r in inputs],
        native=dict(mode='bounded',rounds=3,mfe_weight=.7,optim_num_per_round=800,optim_batchsize=20,optimizer_updates_per_case=120,
            nominal_target_negative_scorer_evaluations_per_case=4800,optimization_proposals_per_case=2400,
            num_seqs_train_gen=2048,num_seqs_select_mfe=512,num_seqs_train_mfe=8192,num_optim_top=80,
            source_sha256=sha(WORK/'external/ribodecode_1_3_0/RiboDecode/train_optim_sim.py'),wrapper_sha256=sha(wrapper),
            weight_sha256=sha(WORK/'external/ribodecode_1_3_0/RiboDecode/score_model/best_model.p'),
            endpoint='Last outer-round native selected sequence; preserve all round outputs; no external-score reselection'),
        own=dict(lambda_value=.75,proposals_per_case=2400,generator='V3 frozen',model='Both continued TE seeds, validation-selected only'),
        fairness='Match nominal proposal count, not gradient updates, scorer calls, unique sequences, folding cost, total compute or runtime. Report these differences.',
        runtime='Native CPU Linux/Python3.8/Torch2.0.1/ViennaRNA2.6.4; all endpoints also folded with common ViennaRNA2.7.2',
        limits=['Finite-budget native algorithm, NOT original default40000x20 run','Three previously used development CDS; lambda0.75 exploratory, no fresh confirmation',
                'HEK293T native cell condition vs human HEK293 scorer; related but not identical conditions',
                'Generated TE and half-life have no experimental labels','Native MFE model is newly trained for each CDS; do not reuse across proteins'],
        source_sha256=sha(__file__))
    if (OUT/'protocol_locked.json').exists():assert read(OUT/'protocol_locked.json')==protocol
    else:save(OUT/'protocol_locked.json',protocol)
    for i,r in enumerate(inputs):
        task=dict(sequence_id='human_'+r['uid'][:16],cds_dna=r['dna'],n_aa=len(r['dna'])//3,seed=20261102+i)
        path=OUT/f'case{i+1}_task.json'
        if path.exists():assert read(path)==task
        else:save(path,task)
    return wrapper,inputs
def main():
    wrapper,rows=prepare();completed=[]
    for i,row in enumerate(rows,1):
        out=OUT/f'case{i}_native';receipt=out/'completion_receipt.json'
        if receipt.exists():
            r=read(receipt);assert r['state']=='completed';completed.append(r);continue
        assert not out.exists(),'Inspect partial native output before retry; never overwrite.'
        cmd=['wsl','-d','Ubuntu','--','/usr/bin/nice','-n','10','/home/lc666/.cache/ds_ribodecode_native/env/bin/python','-u',linux(wrapper),
            '--task',linux(OUT/f'case{i}_task.json'),'--output',linux(out),'--mode','bounded']
        with (OUT/f'case{i}.log').open('xb') as log:
            child=subprocess.Popen(cmd,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save(OUT/'status.json',dict(stage='native_running',case=i,total=3,pid=child.pid,started_unix=time.time(),command=cmd))
            code=child.wait(timeout=4*3600)
        assert code==0,(i,code)
        r=read(receipt);assert r['state']=='completed' and r['scorer_weights_unchanged'] and r['protein_preserved']
        assert r['proposals']==2400 and not r['full_default'];completed.append(r)
        print('native case completed',i,flush=True)
    save(OUT/'native_complete.json',dict(status='completed',cases=3,receipts=[str(OUT/f'case{i}_native/completion_receipt.json') for i in range(1,4)]))
if __name__=='__main__':
    try:main()
    except BaseException as e:save(OUT/'queue_failure.json',dict(error=repr(e),unix=time.time()));raise
