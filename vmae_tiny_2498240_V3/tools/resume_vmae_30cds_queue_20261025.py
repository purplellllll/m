"""Resume the immutable 30-CDS study, keeping complete results and round checkpoints."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from tools import run_vmae_30cds_20261025 as study
OUT=study.OUT


def active_workers():
    found=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            if (directory/'stat').read_text().split()[2]=='Z':continue
            command=(directory/'cmdline').read_bytes().split(b'\0')
            names=[Path(x.decode(errors='replace')).name for x in command if x]
            worker=any(n in ('native_full_gpu_20261025.py','native_full_gpu_resume_20261025.py') for n in names)
            queue='resume_vmae_30cds_queue_20261025.py' in names or ('run_vmae_30cds_20261025.py' in names and b'native' in command)
            if (worker and any(str(OUT).encode() in x for x in command)) or queue:found.append(int(directory.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return found


def completed_round(output,record):
    import torch
    torch.set_num_threads(1)
    candidates=[]
    for path in output.glob('round_*_state.pt'):
        match=re.fullmatch(r'round_(\d+)_state\.pt',path.name)
        if match:candidates.append((int(match.group(1)),path))
    rejected=[]
    for number,path in sorted(candidates,reverse=True):
        try:
            marker=json.loads((output/('round_%02d.json'%number)).read_text())
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['round']==number and 'cuda_rng' in state
            assert state['config']['len_gene_ori']==len(record['cds_dna'])
            assert state['config']['optim_num']==40000 and state['config']['num_optim']==20
            assert marker['native_selected_cds']==state['per_round_selected'][number-1]
            assert set(state['per_round_selected'])==set(range(number))
            del state
            study.save(OUT/'recovery_checkpoints'/(record['sequence_id']+'.json'),dict(
                round=number,checkpoint=str(path),sha256=study.sha(path),rejected_newer_checkpoints=rejected))
            return number
        except (OSError,EOFError,RuntimeError,AssertionError,KeyError) as error:
            rejected.append(dict(path=str(path),error=repr(error)))
    assert not candidates, ('No complete readable checkpoint; inspect without overwriting',rejected)
    return 0


def run_native(record,output,resume_round):
    wrapper=BASE/'tools'/('native_full_gpu_resume_20261025.py' if resume_round else 'native_full_gpu_20261025.py')
    command=[sys.executable,'-u',str(wrapper),'--package',str(OUT/'external/RiboDecode'),
        '--task',str(OUT/'tasks'/(record['sequence_id']+'.json')),'--output',str(output),'--mode','full']
    if resume_round:command.extend(['--resume-round',str(resume_round)])
    with output.with_suffix('.log').open('ab') as log:
        log.write((json.dumps(dict(event='queue_recovery_launch',resume_round=resume_round))+'\n').encode());log.flush()
        child=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
        study.save(OUT/'active_native_process.json',dict(pid=child.pid,task=str(OUT/'tasks'/(record['sequence_id']+'.json')),
            output=str(output),mode='full',resumed_from_completed_round=resume_round))
        assert child.wait()==0, ('Inspect failed recovery worker',str(output))
    return json.loads((output/'completion_receipt.json').read_text())


def main():
    with (OUT/'native_recovery.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not active_workers(), ('Existing active workers',active_workers())
        plan=json.loads((OUT/'plan.json').read_text())
        assert study.sha(BASE/'tools/run_vmae_30cds_20261025.py')==plan['source_sha256']
        assert study.sha(BASE/'tools/native_full_gpu_20261025.py')==plan['native_wrapper_sha256']
        assert all(study.sha(BASE/'latent_decoder'/n)==h for n,h in plan['dependencies'].items())
        validation=json.loads((OUT/'recovery_validation/completion_receipt.json').read_text())
        assert validation['passed'] and validation['max_state_tensor_error']==0
        study.save(OUT/'recovery_plan_20261025.json',dict(original_plan_sha256=study.sha(OUT/'plan.json'),
            source_sha256=study.sha(Path(__file__)),resume_wrapper_sha256=study.sha(BASE/'tools/native_full_gpu_resume_20261025.py'),
            validation=validation,budget='Unchanged20rounds*40000proposals',
            interrupted_partial_round='Preserve and replay from preceding complete round with all RNG states; discarded compute is separate',
            completed_results='Verify and skip; never regenerate completed original/small/native metrics'))
        state=dict(state='recovering',completed=0,total=30)
        study.save(OUT/'native_status.json',state);study.refresh()
        try:
            for record in plan['records']:
                identity=record['sequence_id'];output=OUT/'native'/identity
                receipt_path=OUT/'native_receipts'/(identity+'.json')
                receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                if receipt is None and (output/'completion_receipt.json').exists():
                    receipt=json.loads((output/'completion_receipt.json').read_text())
                if receipt is None:
                    assert shutil.disk_usage(OUT).free>12*2**30,'Insufficient free disk; preserve all results'
                    number=completed_round(output,record) if output.exists() else 0
                    if output.exists() and number==0:
                        target=OUT/'interrupted_workspaces'/(identity+'_'+str(time.time_ns()))
                        target.parent.mkdir(exist_ok=True)
                        assert output.resolve().parent==(OUT/'native').resolve() and target.resolve().is_relative_to(OUT.resolve())
                        output.rename(target)
                    state.update(state='full_native_running',sequence_id=identity,n_aa=record['n_codons']-1,resumed_from_completed_round=number)
                    study.save(OUT/'native_status.json',state);study.refresh()
                    receipt=run_native(record,output,number)
                assert receipt['state']=='completed' and receipt['full_default'] and receipt['rounds']==20
                assert receipt['native_target_negative_queries']==1600000 and receipt['scorer_weights_unchanged']
                study.save(receipt_path,receipt)
                metric_path=OUT/'metrics'/identity/'native.json'
                if not metric_path.exists():
                    subprocess.run([sys.executable,'-u',str(BASE/'tools/run_vmae_30cds_20261025.py'),
                        '--stage','assess-native','--identity',identity],cwd=BASE,check=True)
                metric=json.loads(metric_path.read_text())
                assert metric['cds_dna']==receipt['native_returned_cds']
                archive=OUT/'native_archives'/(identity+'.tar.gz')
                if archive.exists():
                    proof=json.loads(archive.with_suffix('.receipt.json').read_text())
                    assert proof['all_members_verified'] and study.sha(archive)==proof['sha256']
                elif output.exists():study.archive_native(output)
                else:raise AssertionError('Completed workspace and archive both missing')
                state['completed']+=1
                study.save(OUT/'native_status.json',state);study.refresh()
            state.update(state='completed',goal_complete=False)
        except BaseException as error:
            state.update(state='failed',error=repr(error));raise
        finally:
            study.save(OUT/'native_status.json',state);study.refresh()


if __name__=='__main__':main()
