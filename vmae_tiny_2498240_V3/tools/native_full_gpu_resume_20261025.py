"""Recover native CUDA optimization from a completed outer-round boundary."""
import argparse
import ast
import copy
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import types


def validate_history_usage(tree):
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    reads=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='list_seq_gens' and isinstance(n.ctx,ast.Load)]
    assert len(reads)==2,'Audit any new history access before trimming memory'
    for node in reads:
        parent=parents[node]
        if isinstance(parent,ast.Attribute):assert parent.attr=='append'
        else:
            assert isinstance(parent,ast.Subscript)
            assert ast.dump(parent.slice)==ast.dump(ast.parse('a[-5:]').body[0].value.slice)


def trim_history(history):
    del history[:-4]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('parity','preflight','full'),required=True)
    parser.add_argument('--retain-all-history',action='store_true')
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--fold-workers',type=int,default=8)
    parser.add_argument('--resume-round',type=int,default=0)
    args=parser.parse_args()
    assert sys.platform=='linux','The source multiprocessing backend requires Linux fork'
    import resource
    import multiprocessing
    import random
    import numpy as np
    import torch
    import RNA
    assert torch.__version__=='2.5.1+cu124' and RNA.__version__=='2.7.2'
    assert multiprocessing.get_start_method()=='fork'
    assert torch.cuda.is_available() and args.fold_workers>=1
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    task=json.loads(args.task.read_text());output=args.output.resolve()
    assert args.resume_round>=0
    if args.resume_round:
        assert output.is_dir() and not (output/'completion_receipt.json').exists()
        prior_protocol=json.loads((output/'runtime_protocol.json').read_text())
        assert prior_protocol['task']==task and prior_protocol['mode']==args.mode
    else:
        assert not output.exists()
    package=args.package.resolve()
    source=package/'train_optim_sim.py';source_bytes=source.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest()=='bd0150e06fe99ee64b594a8c9d3fd3e03cb04a163133746f75d689f39501eb90'
    tree=ast.parse(source_bytes.decode());validate_history_usage(tree)
    def asset_copy(before,after):
        if Path(before).name=='best_model.p':os.link(before,after);return after
        return shutil.copy2(before,after)
    if not args.resume_round:
        shutil.copytree(str(package),str(output),copy_function=asset_copy)
    config=json.loads((output/'model_config.py').read_text())
    changes={}
    if args.mode=='parity':changes=dict(optim_num=80,optim_batchsize=1,num_workers=0)
    if args.mode=='preflight':changes=dict(optim_num=80,optim_batchsize=20,num_seqs_train_gen=128,num_seqs_select_mfe=16,num_seqs_train_mfe=64,num_optim_top=20)
    if changes and not args.resume_round:
        config['training'].update(changes);(output/'model_config.py').write_text(json.dumps(config,indent=2))
    rounds={'parity':1,'preflight':2,'full':20}[args.mode]
    weight=0. if args.mode=='parity' else .7
    weights=package/'score_model/best_model.p';weight_sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    run_config=config['training'];budget=run_config['optim_num'];batch=run_config['optim_batchsize']
    assert args.resume_round<rounds
    if args.mode=='full':assert not changes and (budget,batch,rounds,weight)==(40000,20,20,.7)
    previous_elapsed=0.
    if args.resume_round:
        observations=[json.loads(s) for s in (output/'events.jsonl').read_text().splitlines() if s.strip()]
        previous_elapsed=max(r['seconds'] for r in observations)
    context=dict(phase='setup',round=args.resume_round,queries=2*budget*args.resume_round,fold_sequences=0,fold_calls=0,max_history=0)
    if args.resume_round:
        assert run_config['sampler_type']=='dist-optim'
        top=min(run_config['num_optim_top'],3*budget//4)
        generated=run_config['num_seqs_select_mfe']+(args.resume_round-1)*2*(run_config['num_seqs_select_mfe']//4) if weight>0 else 0
        context['fold_sequences']=generated+args.resume_round*top
        context['fold_calls']=generated*math.ceil(len(task['cds_dna'])/run_config['mfe_tool_sub_len'])+args.resume_round*top
    def event(kind,**values):
        record=dict(event=kind,seconds=previous_elapsed+time.monotonic()-started,phase=context['phase'],round=context['round'],
            native_target_negative_queries=context['queries'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**values)
        line=json.dumps(record,allow_nan=False)
        with (output/'events.jsonl').open('a') as handle:handle.write(line+'\n')
        print(line,flush=True)
    def write(name,value):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    started=time.monotonic()
    protocol_name='resume_protocol_round_%02d_%d.json'%(args.resume_round,time.time_ns()) if args.resume_round else 'runtime_protocol.json'
    write(protocol_name,dict(mode=args.mode,task=task,seed=task['seed'],overrides=changes,outer_rounds=rounds,mfe_weight=weight,
        resumed_from_completed_round=args.resume_round,observed_prior_elapsed_seconds=previous_elapsed,
        discarded_partial_round_compute_not_in_nominal_query_and_fold_counts=bool(args.resume_round),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=sys.platform,python=sys.version,torch=torch.__version__,viennarna=RNA.__version__,numpy=np.__version__,
        memory_history='all' if args.retain_all_history else 'Keep last5 tensors; original source reads only last5, statically checked',
        cpu_resources=dict(torch_threads=4,fold_workers=args.fold_workers,data_loader_workers=run_config['num_workers']),
        gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,tf32=False,cudnn_deterministic=True,
        numerical_changes=False,algorithm='Shipped source statements executed in order, with logging and unobserved history trimming only',
        reference='Last outer-round selected sequence, also written by native source to data/cds/optim_data.txt; retain all per-round selections'))
    sys.path.insert(0,str(package.parent))
    from RiboDecode.score_model.inference import InferenceModel_conditon_spec
    original_predict=InferenceModel_conditon_spec.predict_seq_spec_single
    def counted(self,sequences,*positional,**keyword):
        context['queries']+=2*len(sequences)
        return original_predict(self,sequences,*positional,**keyword)
    InferenceModel_conditon_spec.predict_seq_spec_single=counted
    original_copytree=shutil.copytree
    def isolated_copy(before,after,*positional,**keyword):
        if Path(before).resolve()==package and Path(after).resolve()==output:return str(output)
        return original_copytree(before,after,*positional,**keyword)
    shutil.copytree=isolated_copy
    torch.set_num_threads(4)
    random.seed(task['seed']);np.random.seed(task['seed']);torch.manual_seed(task['seed'])
    torch.cuda.manual_seed_all(task['seed'])
    old_main=sys.modules['__main__'];native=types.ModuleType('__main__');native.__file__=str(source)
    sys.modules['__main__']=native;namespace=native.__dict__
    os.chdir(output)
    sys.argv=[str(source),'--cds',task['sequence_id'],'--cds_seq',task['cds_dna'],'--env','HEK293T',
        '--alpha','100','--beta','100','--mfe_weight',str(weight),'--optim_epoch',str(rounds)]
    split=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parser' for t in node.targets))
    try:
        exec(compile(ast.Module(body=tree.body[:split],type_ignores=[]),str(source),'exec'),namespace)
        namespace['cpu_count']=lambda:args.fold_workers+1
        original_generate=namespace['seq_codon_gen'].generate
        def generate(self,codon_mask):
            history=sys._getframe(1).f_locals.get('list_seq_gens')
            if history is not None:
                if not args.retain_all_history:trim_history(history)
                context['max_history']=max(context['max_history'],len(history)+1)
            return original_generate(self,codon_mask)
        namespace['seq_codon_gen'].generate=generate
        original_iterator=torch.utils.data.DataLoader.__iter__
        def iterator(loader):
            last=time.monotonic()
            for index,data in enumerate(original_iterator(loader),1):
                if time.monotonic()-last>=30:
                    event('batches',yielded=index,total=len(loader),batch_size=loader.batch_size);last=time.monotonic()
                yield data
        torch.utils.data.DataLoader.__iter__=iterator
        for function in ('get_mfe','get_mfe_sim'):
            def wrap_fold(original,chunked):
                @functools.wraps(original)
                def fold(items,*positional,**keyword):
                    factor=math.ceil(len(task['cds_dna'])/(keyword.get('len_sub',4000))) if chunked else 1
                    context['fold_sequences']+=len(items);context['fold_calls']+=len(items)*factor
                    return original(items,*positional,**keyword)
                return fold
            namespace[function]=wrap_fold(namespace[function],function=='get_mfe')
        def wrap_phase(original,name):
            @functools.wraps(original)
            def phase(*positional,**keyword):
                context['phase']=name;context['round']=namespace['idx_optim']+1;clock=time.monotonic();event('phase_start')
                value=original(*positional,**keyword);native_seconds=time.monotonic()-clock
                if name=='optim':
                    index=namespace['idx_optim'];results=namespace['data_gen_optim']
                    selected=results['seq_optim_best'][index]
                    assert (output/'data/cds/optim_data.txt').read_text().strip()==selected
                    write('round_%02d.json'%context['round'],dict(round=context['round'],native_selected_cds=selected,
                        native_rpf=results['results_rpf'][index],native_mfe=results['results_mfe'][index],
                        source_seconds=native_seconds,cumulative_queries=context['queries']))
                    torch.save(dict(z=namespace['z_model'].state_dict(),mfe=namespace['model_mfe'].state_dict(),
                        round=context['round'],config=namespace['model_config'],python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(),previous_sequences=results['seqs_optim_gen'][index],previous_distribution=results['mask_optim_dist'][index],
                        per_round_selected=results['seq_optim_best'],per_round_rpf=results['results_rpf'],per_round_mfe=results['results_mfe']),output/('round_%02d_state.pt'%context['round']))
                event('phase_end',source_seconds=native_seconds);return value
            return phase
        for name in ('gen_train','mfe_train','optim'):namespace[name]=wrap_phase(namespace[name],name)
        event('started')
        if args.resume_round:
            tail=tree.body[split:]
            loops=[i for i,n in enumerate(tail) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='idx_optim']
            assert len(loops)==1
            loop_index=loops[0]
            exec(compile(ast.Module(body=tail[:loop_index],type_ignores=[]),str(source),'exec'),namespace)
            checkpoint=output/('round_%02d_state.pt'%args.resume_round)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['round']==args.resume_round and 'cuda_rng' in state
            assert all(state['config'][k]==v for k,v in namespace['model_config'].items() if k!='data_folder')
            assert set(state['per_round_selected'])==set(range(args.resume_round))
            preserved=output/'interrupted'/('after_round_%02d_%d'%(args.resume_round,time.time_ns()))
            preserved.mkdir(parents=True,exist_ok=False)
            source_results=output/'results_natural/HEK293T_optim_mfe_dist-optim'
            for directory in source_results.iterdir():
                if directory.is_dir() and directory.name.isdigit() and int(directory.name)>=args.resume_round:
                    assert directory.resolve().is_relative_to(output)
                    directory.rename(preserved/('results_round_'+directory.name))
            for file in output.glob('round_*'):
                if file.is_file() and int(file.name.split('_')[1].split('.')[0])>args.resume_round:
                    file.rename(preserved/file.name)
            namespace['z_model'].load_state_dict(state['z'],strict=True)
            namespace['model_mfe'].load_state_dict(state['mfe'],strict=True)
            results=namespace['data_gen_optim'];index=args.resume_round-1
            results['seqs_optim_gen'][index]=state['previous_sequences']
            results['mask_optim_dist'][index]=state['previous_distribution']
            results['seq_optim_best'].update(state['per_round_selected'])
            results['results_rpf'].update(state['per_round_rpf'])
            results['results_mfe'].update(state['per_round_mfe'])
            target=output/'data/cds/optim_data.txt'
            selected=state['per_round_selected'][index]
            if target.read_text().strip()!=selected:
                target.rename(preserved/'optim_data_before_resume.txt')
                target.write_text(selected+'\n')
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            write('resume_checkpoint_round_%02d.json'%args.resume_round,dict(
                checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                completed_round=args.resume_round,next_round=args.resume_round+1,
                python_numpy_torch_cpu_cuda_rng_restored=True,partial_outputs_preserved=str(preserved)))
            event('checkpoint_restored',next_round=args.resume_round+1)
            loop=copy.deepcopy(tail[loop_index])
            assert ast.dump(loop.iter)==ast.dump(ast.parse('range(num_optim)',mode='eval').body)
            loop.iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(args.resume_round),ast.Name(id='num_optim',ctx=ast.Load())],keywords=[])
            resumed=ast.fix_missing_locations(ast.Module(body=[loop]+tail[loop_index+1:],type_ignores=[]))
            exec(compile(resumed,str(source),'exec'),namespace)
        else:
            exec(compile(ast.Module(body=tree.body[split:],type_ignores=[]),str(source),'exec'),namespace)
        result=namespace['data_gen_optim'];selected=result['seq_optim_best'][rounds-1]
        assert context['queries']==2*budget*rounds
        assert len(selected)==len(task['cds_dna']) and set(selected)<=set('ACGT')
        check=namespace['check'];assert all(check.search(selected[i:i+3])==check.search(task['cds_dna'][i:i+3]) for i in range(0,len(selected),3))
        assert hashlib.sha256(weights.read_bytes()).hexdigest()==weight_sha
        receipt=dict(state='completed',mode=args.mode,full_default=args.mode=='full',rounds=rounds,proposals=budget*rounds,
            optimizer_updates=budget//batch*rounds,native_target_negative_queries=context['queries'],
            true_fold_sequence_evaluations=context['fold_sequences'],true_fold_calls=context['fold_calls'],
            elapsed_seconds=previous_elapsed+time.monotonic()-started,max_retained_soft_batches=context['max_history'],
            resumed_from_completed_round=args.resume_round,observed_prior_elapsed_seconds=previous_elapsed,
            nominal_queries_exclude_discarded_partial_round_work=bool(args.resume_round),
            native_returned_cds=selected,all_round_selected=result['seq_optim_best'],all_round_native_rpf=result['results_rpf'],
            scorer_sha256=weight_sha,scorer_weights_unchanged=True,source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            stop_preserved=selected[-3:]==task['cds_dna'][-3:],protein_preserved=True,goal_complete=False)
        write('completion_receipt.json',receipt);event('completed',full_default=receipt['full_default'])
    except BaseException as error:
        write('failure.json',dict(state='failed',error=repr(error),context=context));raise
    finally:
        shutil.copytree=original_copytree;InferenceModel_conditon_spec.predict_seq_spec_single=original_predict
        if 'original_iterator' in locals():torch.utils.data.DataLoader.__iter__=original_iterator
        sys.modules['__main__']=old_main


if __name__=='__main__':main()
