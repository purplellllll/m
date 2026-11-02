"""After TE continuation, search the same CDS at the native nominal proposal budget."""
from pathlib import Path
import sys,json,hashlib,time,importlib.util
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent.parent;OUT=HERE/'native_compare/own_search'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(WORK))
from stability import read,save,sha,StabilityScorer
from latent_decoder.model import CodonVMAE,collate_records,mutable_positions,synonymous_logits,MASK,AA_IDS
from latent_decoder.tokenizer import CODONS
from latent_decoder.prepare_data import GENETIC_CODE
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'scores').mkdir(exist_ok=True)
    deadline=time.monotonic()+7200
    while not (HERE/'te_24000/training_complete.json').exists():
        if (HERE/'te_24000/training_failure.json').exists():raise RuntimeError(read(HERE/'te_24000/training_failure.json'))
        if time.monotonic()>deadline:raise TimeoutError('TE continuation two-hour deadline')
        time.sleep(5)
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    text=(HERE/'locked_scorer.py').read_text(encoding='utf-8')
    assert "ROOT/'te_continuation'" in text
    text=text.replace('te_continuation','te_24000')
    text=text.replace('18000steps per seed','up to24000steps per seed, validation-selected')
    path=HERE/'locked_scorer_24000.py'
    if path.exists():assert path.read_text(encoding='utf-8')==text
    else:path.write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('new_continued_scorer',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    scorer=m.LockedScorer('cuda');stability=StabilityScorer()
    selected=read(HERE/'te_24000/selection_locked.json')['selected_models']
    assert [r[3] for r in scorer.models]==[r['checkpoint_sha256'] for r in selected]
    # Check both trained heads against stored validation predictions before any candidate search.
    validation=[r for r in read(HERE/'te_24000/train_validation.json') if r['split']=='val'];errors=[]
    for index in [0,11,49]:
        value=scorer(validation[index]['dna'])
        errors.extend(abs(a-b[2]['validation_predictions'][index]) for a,b in zip(value['seed_scores'],scorer.models))
    assert max(errors)<2e-5
    save(OUT/'reload_check.json',dict(status='passed',max_error=max(errors),checkpoint_sha256=[r[3] for r in scorer.models]))
    rows=read(HERE/'lambda_scan/inputs_locked.json');native=read(HERE/'native_compare/protocol_locked.json')
    steps=native['own']['proposals_per_case'];assert steps==2400
    generator=WORK/'runs/vmae_tiny_size_only_20261025/checkpoints/fold0_uniform.pt'
    payload=torch.load(generator,map_location='cpu',weights_only=False)
    model=CodonVMAE(**payload['config']['model']).cuda().eval().requires_grad_(False);model.load_state_dict(payload['model'],strict=True)
    yscale=float(scorer.models[0][2]['target_scale']);sscale=float(np.std([r['half_life_log_target'] for r in read(HERE/'stability/development_records.json') if r['split']=='train']))
    protocol=dict(lambda_value=.75,proposals_per_case=steps,cases=[r['uid'] for r in rows],
        own_checkpoint_sha256=[r[3] for r in scorer.models],generator_sha256=sha(generator),stability_model_sha256=sha(HERE/'stability/model.npz'),
        source_sha256=sha(__file__),scorer_source_sha256=sha(path),native_protocol_sha256=sha(HERE/'native_compare/protocol_locked.json'),
        scales=dict(own=yscale,stability=sscale),objective='Same frozen lambda0.75 and GC/CAI/MFE/half-life combination as prior development scan',
        model_selection='Validation-best continued checkpoints; no903 test opened',
        fairness='2400 nominal proposals per CDS, same as bounded native; gradient updates, folds, unique queries and total compute differ')
    if (OUT/'protocol_locked.json').exists():assert read(OUT/'protocol_locked.json')==protocol
    else:save(OUT/'protocol_locked.json',protocol)
    def score(dna,row):
        uid=hashlib.sha256(dna.encode()).hexdigest();p=OUT/'scores'/f'{uid}.json'
        if p.exists():v=read(p);assert v['checkpoint_sha256']==protocol['own_checkpoint_sha256']
        else:
            v=scorer(dna);save(p,v)
        return dict(**v,stability_log_score=stability(dna,row['utr5'],row['utr3']))
    for ci,row in enumerate(rows):
        result=OUT/f'case{ci+1}.json'
        if result.exists():continue
        begun=time.monotonic();dna=row['dna'];original=score(dna,row);best=original;best_reward=0.;trace=[]
        tokens=collate_records([dict(cds_dna=dna)]).cuda();active=mutable_positions(tokens)
        with torch.no_grad():mu,lv=model.encode(tokens.masked_fill(active,MASK),tokens);sigma=(lv*.5).exp()
        offset=torch.zeros_like(mu);latent_best=-float('inf');rng=torch.Generator(device='cuda').manual_seed(20261102+ci)
        def reward(v):
            t=(v['own_score']-original['own_score'])/yscale
            p=np.mean([(v['gc_percent']-original['gc_percent'])/5.,(v['cai']-original['cai'])/.05,
                (original['mfe_per_nt']-v['mfe_per_nt'])/.05,(v['stability_log_score']-original['stability_log_score'])/sscale])
            return float(.75*t+.25*p)
        for step in range(steps):
            with torch.no_grad():
                trial=offset if step==0 else offset+torch.randn(mu.shape,generator=rng,device='cuda')*active[...,None]*max(.25,.95**(step//10))
                trial=trial*active[...,None];rms=trial[active].square().mean().sqrt();trial*=min(1.,8/float(rms.clamp_min(1e-12)))
                ids=synonymous_logits(model.decode(mu+sigma*trial,tokens),tokens).argmax(-1);ids[~active]=tokens[~active]
                assert torch.equal(AA_IDS.to(ids.device)[ids],AA_IDS.to(ids.device)[tokens]);candidate=''.join(CODONS[int(i)] for i in ids[0])
            assert candidate[:3]==dna[:3] and candidate[-3:]==dna[-3:] and len(candidate)==len(dna)
            assert all(GENETIC_CODE[candidate[j:j+3]]==GENETIC_CODE[dna[j:j+3]] for j in range(0,len(dna),3))
            v=score(candidate,row);r=reward(v);trace.append(dict(step=step,sequence_sha256=v['sequence_sha256'],reward=r))
            if r>latent_best:latent_best=r;offset=trial.clone()
            if r>best_reward:best_reward=r;best=v
            if (step+1)%20==0:
                save(OUT/'status.json',dict(stage='own_search',case=ci+1,step=step+1,total_steps=steps,best_reward=best_reward))
                if (step+1)%200==0:print('own search',ci+1,step+1,steps,best_reward,flush=True)
        assert abs(best_reward-max([0.]+[r['reward'] for r in trace]))<1e-10
        save(result,dict(case_uid=row['uid'],original=original,best=best,best_reward=best_reward,trace=trace,
            attempts=steps,unique_proposals=len({r['sequence_sha256'] for r in trace}),seconds=time.monotonic()-begun))
    save(OUT/'complete.json',dict(status='completed',cases=3,proposals=7200,lambda_value=.75))
if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True);save(OUT/'failure.json',dict(error=repr(error),unix=time.time()));raise
