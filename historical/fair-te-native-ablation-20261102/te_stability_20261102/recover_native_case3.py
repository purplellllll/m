"""Replay failed case3 on Linux ext4; preserve failed mounted-drive attempt."""
from pathlib import Path
import json,sys,time,subprocess,hashlib,shutil,os
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;OUT=HERE/'native_compare'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')
def linux(p):
    p=Path(p).resolve();return '/mnt/'+p.drive[0].lower()+'/'+p.relative_to(p.anchor).as_posix()
def linux_main():
    import hashlib
    root=Path('/home/lc666/.cache/te_native_20261102_case3_retry1')
    assert not root.exists(), 'Inspect existing recovery before retry'
    root.mkdir(parents=True)
    mounted=Path(sys.argv[sys.argv.index('--linux')+1])
    base=mounted.parent.parent.parent
    original=base/'external/ribodecode_1_3_0/RiboDecode'
    package=root/'RiboDecode';shutil.copytree(original,package)
    h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name in ['train_optim_sim.py','score_model/best_model.p']:
        assert h(package/name)==h(original/name)
    source=(mounted/'native_bounded_cpu.py').read_text(encoding='utf-8')
    old='package=Path('+repr(str(original))+')';assert source.count(old)==1
    source=source.replace(old,'package=Path('+repr(str(package))+')')
    wrapper=root/'native_bounded_cpu.py';wrapper.write_text(source,encoding='utf-8')
    task=root/'task.json';shutil.copy2(mounted/'case3_task.json',task)
    with (root/'run.log').open('wb') as log:
        result=subprocess.run([sys.executable,'-u',str(wrapper),'--task',str(task),'--output',str(root/'case3_native'),'--mode','bounded'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0, ('Inspect Linux log',str(root/'run.log'),result.returncode)
    receipt=read(root/'case3_native/completion_receipt.json');assert receipt['state']=='completed' and receipt['proposals']==2400
    target=mounted/'case3_native';assert not target.exists()
    def copyfile(src,dst):
        if Path(src).name=='best_model.p':os.link(str(original/'score_model/best_model.p'),dst);return dst
        return shutil.copy2(src,dst)
    shutil.copytree(root/'case3_native',target,copy_function=copyfile)
    shutil.copy2(root/'run.log',mounted/'case3_retry_ext4.log')
    return dict(status='completed',storage='Linux ext4',linux_root=str(root),same_seed=True,
        restart_from='Original case3 input and seed; not a partial numerical resume',source_sha256=h(package/'train_optim_sim.py'),
        wrapper_sha256=h(wrapper),original_weights_sha256=h(package/'score_model/best_model.p'))
def main():
    if '--linux' in sys.argv:
        record=linux_main();print(json.dumps(record),flush=True);return
    old=OUT/'case3_native';archive=OUT/'case3_native_interrupted_io_1'
    assert old.resolve().is_relative_to(OUT.resolve()) and archive.resolve().is_relative_to(OUT.resolve())
    assert read(old/'failure.json')['error'].startswith('OSError(5,')
    failed=read(old/'failure.json');assert not archive.exists();old.rename(archive)
    for name in ['queue_failure.json','comparison_failure.json']:
        path=OUT/name
        if path.exists():path.rename(OUT/(name+'.archived_io_1'))
    save(OUT/'recovery_status.json',dict(stage='native_case3_replay_on_ext4',linux_root='/home/lc666/.cache/te_native_20261102_case3_retry1',failed_attempt=failed))
    command=['wsl','-d','Ubuntu','--','/usr/bin/nice','-n','10','/home/lc666/.cache/ds_ribodecode_native/env/bin/python','-u',linux(Path(__file__)),'--linux',linux(OUT)]
    with (OUT/'case3_recovery_driver.log').open('xb') as log:
        result=subprocess.run(command,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0, result.returncode
    for i in [1,2,3]:assert read(OUT/f'case{i}_native/completion_receipt.json')['state']=='completed'
    save(OUT/'case3_recovery_execution.json',dict(status='completed',same_case_seed_budget=True,
        failed_attempt_context=failed['context'],discarded_work_is_extra=True,linux_root='/home/lc666/.cache/te_native_20261102_case3_retry1'))
    save(OUT/'native_complete.json',dict(status='completed',cases=3,case3_replayed_after_io_failure=True))
    print('Native3-case queue recovered and completed',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--linux' not in sys.argv:save(OUT/'queue_failure.json',dict(error=repr(error),unix=time.time()))
        raise
