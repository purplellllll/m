"""V4 fixed-split fine-tuning from frozen V3 with monotonic GC/CAI/exact-MFE rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path('/root/gpufree-data/vmae_transfer_assessment_20261024')
OUT = BASE.parent/'vmae_v4_rewards_20261026'
CHECKPOINT = BASE.parent/'vmae_tiny_size_only_20261025/results/fold_0/uniform/best.pt'
CHECKPOINT_SHA = '6592a07293da25b18f9ed71584e48dc4da23fa121b4f735f10d26ee7da26c6af'
SEED = 20261026


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def assert_split(data):
    groups = {p: {r['v2_group'] for r in data[p]} for p in ('train','val','test')}
    assert all(not groups[a] & groups[b] for a,b in (('train','val'),('train','test'),('val','test')))
    ids = {p: {r['sequence_id'] for r in data[p]} for p in groups}
    assert all(not ids[a] & ids[b] for a,b in (('train','val'),('train','test'),('val','test')))


def selfcheck():
    data = {p:[dict(v2_group=p,sequence_id=p)] for p in ('train','val','test')}
    assert_split(data)
    data['val'][0]['v2_group']='train'
    try:
        assert_split(data)
    except AssertionError:
        return
    raise AssertionError('Split leakage was accepted')


def main(preflight=False):
    import fcntl
    import torch
    from torch.nn import functional as F
    from latent_decoder.model import collate_records, corrupt, mutable_positions, MASK, vae_loss
    from latent_decoder.tokenizer import CODONS
    from latent_decoder.train import codon_frequency
    from latent_decoder.v2_train import corpus, atomic_torch
    from latent_decoder.v2_data import pick
    from latent_decoder.latent_information_probe import load_model
    from latent_decoder.latent_optimization import TranslationObjective, state_hash, design_probabilities
    from latent_decoder.dual_space_ablation import CountedScorer
    from tools.vmae_v4_rewards_20261026 import property_rewards, mfe_reward, exact_fold, scst_loss

    assert torch.cuda.is_available()
    assert digest(CHECKPOINT)==CHECKPOINT_SHA
    OUT.mkdir(parents=True,exist_ok=True)
    handle=(OUT/'training.lock').open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    torch.set_num_threads(4)
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data_dir=BASE.parent/'vmae_adjustment_20261024/data'
    data=corpus(data_dir,0)
    assert_split(data)
    validation=pick(data['val'],8,'multiobjective-val-20261026')
    assert len(validation)==24
    counts=codon_frequency(data['train'])
    from latent_decoder.tokenizer import AA
    weights={c:float(counts[i]/max(counts[j] for j,a in enumerate(AA) if a==AA[i])) for i,c in enumerate(CODONS)}
    model=load_model(CHECKPOINT,'cuda').requires_grad_(True)
    assert sum(p.numel() for p in model.parameters())==2498240
    initial_model_hash=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    scorer=CountedScorer(TranslationObjective(BASE/'score_model')).cuda().eval().requires_grad_(False)
    canonical=BASE.parent/'vmae_transfer_20261024/canonical_condition.pt'
    subset=json.loads((BASE/'scorer_subset_receipt.json').read_text())
    assert digest(canonical)==subset['canonical_condition_sha256']
    with torch.no_grad():
        scorer.scorer.condition.copy_(torch.load(canonical,map_location='cpu',weights_only=True).cuda())
    assert state_hash(scorer.scorer)==subset['model_state_hash']
    scorer_hash=state_hash(scorer)

    def generated(tokens):
        inputs=tokens.masked_fill(mutable_positions(tokens),MASK)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=model.training):
            mu,_=model.encode(inputs,tokens)
            logits=model.decode(mu,tokens)
        return design_probabilities(logits.float(),tokens)

    if preflight:
        assert torch.cuda.is_bf16_supported()
        model.train()
        records=sorted(data['train'],key=lambda r:r['n_codons'])[-4:]
        tokens=collate_records(records).cuda()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        corrupted,selected=corrupt(tokens,.15)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,mu,lv=model(corrupted,tokens)
        reconstruction,_=vae_loss(logits.float(),tokens,mu.float(),lv.float(),selected,beta=.001)
        p=generated(tokens)
        hard=F.one_hot(p.argmax(-1),64).float()
        score=scorer(hard+p-p.detach(),tokens).mean()
        props=property_rewards(hard+p-p.detach(),tokens,weights)
        cache={}
        last=len(records)-1
        length=records[last]['n_codons']
        original=exact_fold(records[last]['cds_dna'],cache)['mfe_per_nt']
        scst,feedback=scst_loss(p[last:last+1,:length],tokens[last:last+1,:length],original,cache)
        loss=reconstruction+.1*(-score-props['gc_reward'].mean()-props['cai_reward'].mean()+16*scst)
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert finite and float(norm)>0
        assert state_hash(scorer)==scorer_hash and digest(CHECKPOINT)==CHECKPOINT_SHA
        save(OUT/'preflight.json',dict(passed=True,batch_size=len(records),
            lengths=[r['n_codons'] for r in records],peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            loss=float(loss),finite_grad=finite,gradient_norm_before_clip=float(norm),
            raw_scst_loss=float(scst.detach()),mfe_feedback=feedback,
            optimizer_step_performed=False,original_checkpoint_unchanged=True,
            elapsed_seconds=time.time()-started))
        return
    assert torch.cuda.is_bf16_supported()
    plan=dict(version='V4',reward_mode='Unclipped monotonic increases in GC, CAI and negative MFE/nt',seed=SEED,epochs=3,batch_size=4,learning_rate=1e-5,gradient_clip=1.,beta=.001,
        precision='BF16 autocast model forward; FP32 parameters, optimizer, losses and scoring; FP32 validation',
        gradient_accumulation=1,train_coverage='Every fold-0 training CDS once per epoch',
        vocabulary=CODONS,scorer_checkpoint_sha256=digest(BASE/'score_model/best_model.p'),
        property_weight=.1,mfe_weight=1.,mfe_every_microbatches=16,
        mfe_frequency_correction=16,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
        fold=0,train_cases=len(data['train']),val_ids=[r['sequence_id'] for r in validation],
        test_used=False,selection='24 fixed validation CDS; reconstruction plus greedy multiobjective loss',
        gc_reward_scale=.05,cai_reward_scale=.05,mfe_per_nt_reward_scale=.05,
        reward_weights=dict(gc=1.,cai=1.,mfe=1.),reward_reference='Parent CDS, centering only; no threshold or saturation',
        training_loss='CE_mask+CE_full+0.001*KL-0.1*(Hard+R_GC+R_CAI)+0.1*16*SCST_MFE_loss',
        canonical_condition_sha256=digest(canonical),scorer_hash=scorer_hash,
        source_sha256={str(p.relative_to(BASE)):digest(p) for p in (
            Path(__file__).resolve(),BASE/'tools/vmae_v4_rewards_20261026.py',BASE/'tools/vmae_multiobjective_core_20261026.py',
            BASE/'latent_decoder/model.py',BASE/'latent_decoder/tokenizer.py',
            BASE/'latent_decoder/latent_optimization.py',BASE/'latent_decoder/v2_train.py',
            BASE/'latent_decoder/latent_information_probe.py')},
        corpus_sha256=digest(data_dir/'corpus.jsonl.gz'),folds_sha256=digest(data_dir/'folds.json'))
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text())==plan,'Resume plan differs'
    else:
        save(OUT/'plan.json',plan); save(OUT/'cai_weights.json',weights)
    cache=json.loads((OUT/'fold_cache.json').read_text()) if (OUT/'fold_cache.json').exists() else {}
    initial=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    epoch_start,step,best=0,0,float('inf')
    if (OUT/'last.pt').exists():
        state=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        epoch_start,step,best=state['epoch'],state['step'],state['best']
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
    if (OUT/'complete.json').exists():
        assert digest(CHECKPOINT)==CHECKPOINT_SHA
        return
    save(OUT/'status.json',dict(state='training',started_unix=started,heartbeat_unix=time.time(),
        completed_epochs=epoch_start,total_epochs=3,step=step))

    def original_fold(record):
        return exact_fold(record['cds_dna'],cache)['mfe_per_nt']

    def dna_from(tokens):
        return ''.join(CODONS[i] for i in tokens.tolist())

    @torch.no_grad()
    def evaluate():
        model.eval()
        results=[]
        # Validation masking does not alter the training RNG stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for index,r in enumerate(validation):
                torch.manual_seed(SEED+index)
                tokens=collate_records([r]).cuda()
                masked,selected=corrupt(tokens,.15)
                if not selected.any():
                    positions=mutable_positions(tokens).nonzero()
                    assert len(positions), 'Validation CDS has no mutable codons'
                    selected[tuple(positions[0])]=True
                    masked=tokens.masked_fill(selected,MASK)
                mu,lv=model.encode(masked,tokens)
                reconstruction,_=vae_loss(model.decode(mu,tokens),tokens,mu,lv,selected,beta=.001)
                p=generated(tokens); chosen=p.argmax(-1)
                hard=F.one_hot(chosen,64).float()
                score=scorer(hard,tokens).mean()
                props=property_rewards(hard,tokens,weights)
                folded=exact_fold(dna_from(chosen[0]),cache)
                mr=float(mfe_reward(folded['mfe_per_nt'],original_fold(r)))
                objective=reconstruction+.1*(-score-props['gc_reward'].mean()-props['cai_reward'].mean()-mr)
                results.append(dict(sequence_id=r['sequence_id'],objective=float(objective),
                    reconstruction=float(reconstruction),hard_score=float(score),
                    gc_reward=float(props['gc_reward'].mean()),cai_reward=float(props['cai_reward'].mean()),
                    mfe_reward=mr,mfe_per_nt=folded['mfe_per_nt']))
        return dict(objective=sum(r['objective'] for r in results)/len(results),rows=results)

    if epoch_start==0:
        save(OUT/'status.json',dict(state='baseline_validation',started_unix=started,
            heartbeat_unix=time.time(),completed_epochs=0,total_epochs=3,step=step))
        baseline=evaluate()
        best=baseline['objective']
        assert torch.isfinite(torch.tensor(best))
        save(OUT/'baseline_validation.json',baseline)
        baseline_checkpoint=dict(model=model.state_dict(),config=initial['config'],epoch=0,objective=best)
        if 'recipe' in initial: baseline_checkpoint['recipe']=initial['recipe']
        atomic_torch(OUT/'best.pt',baseline_checkpoint)
        save(OUT/'fold_cache.json',cache)
    for epoch in range(epoch_start,3):
        model.train()
        order=torch.randperm(len(data['train'])).tolist()
        sums=dict(loss=0.,reconstruction=0.,hard_score=0.,gc_reward=0.,cai_reward=0.,
            raw_scst_loss=0.,frequency_corrected_mfe_loss=0.)
        samples=0; feedback=0
        for offset in range(0,len(order),4):
            records=[data['train'][i] for i in order[offset:offset+4]]
            tokens=collate_records(records).cuda()
            masked,selected=corrupt(tokens,.15)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,mu,lv=model(masked,tokens)
            rec,_=vae_loss(logits.float(),tokens,mu.float(),lv.float(),selected,beta=.001)
            p=generated(tokens)
            hard=F.one_hot(p.argmax(-1),64).float()
            score=scorer(hard+p-p.detach(),tokens).mean()
            props=property_rewards(hard+p-p.detach(),tokens,weights)
            scst=rec.new_zeros(())
            if (step+1)%16==0:
                chosen=int(torch.randint(len(records),()).item())
                length=records[chosen]['n_codons']
                scst,diagnostics=scst_loss(p[chosen:chosen+1,:length],tokens[chosen:chosen+1,:length],
                    original_fold(records[chosen]),cache)
                feedback+=1
            weighted_mfe=16*scst
            loss=rec+.1*(-score-props['gc_reward'].mean()-props['cai_reward'].mean()+weighted_mfe)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); step+=1
            if step==1:
                updated_hash=state_hash(model)
                assert updated_hash!=initial_model_hash and state_hash(scorer)==scorer_hash
                save(OUT/'first_update.json',dict(version='V4',step=step,loss=float(loss.detach()),
                    gc_reward=float(props['gc_reward'].mean().detach()),cai_reward=float(props['cai_reward'].mean().detach()),
                    model_hash_before=initial_model_hash,model_hash_after=updated_hash,
                    scorer_unchanged=True,source_checkpoint_unchanged=digest(CHECKPOINT)==CHECKPOINT_SHA,
                    recorded_unix=time.time()))
            terms=dict(loss=loss,reconstruction=rec,hard_score=score,gc_reward=props['gc_reward'].mean(),
                cai_reward=props['cai_reward'].mean(),raw_scst_loss=scst,
                frequency_corrected_mfe_loss=weighted_mfe)
            for key,value in terms.items(): sums[key]+=float(value.detach())*len(records)
            samples+=len(records)
            if step%16==0:
                save(OUT/'progress.json',dict(epoch=epoch+1,step=step,samples=samples,mfe_feedback=feedback,
                    elapsed_seconds=time.time()-started,last_mfe_feedback=diagnostics,
                    means={k:v/samples for k,v in sums.items()}))
                save(OUT/'status.json',dict(state='training',started_unix=started,heartbeat_unix=time.time(),
                    elapsed_seconds=time.time()-started,completed_epochs=epoch,total_epochs=3,
                    current_epoch=epoch+1,step=step,epoch_samples=samples,mfe_feedback=feedback))
        assert samples==len(data['train'])
        save(OUT/'status.json',dict(state='validating',started_unix=started,heartbeat_unix=time.time(),
            elapsed_seconds=time.time()-started,completed_epochs=epoch,total_epochs=3,current_epoch=epoch+1,step=step))
        validation_result=evaluate()
        value=validation_result['objective']; assert torch.isfinite(torch.tensor(value))
        improved=value<best; best=min(best,value)
        assert digest(CHECKPOINT)==CHECKPOINT_SHA and state_hash(scorer)==scorer_hash
        state=dict(model=model.state_dict(),config=initial['config'],epoch=epoch+1,step=step,best=best,
            optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate())
        if 'recipe' in initial: state['recipe']=initial['recipe']
        atomic_torch(OUT/'last.pt',state)
        if improved:
            atomic_torch(OUT/'best.pt',dict(
                **{k:v for k,v in state.items() if k in ('model','config','recipe','epoch')},objective=value))
        save(OUT/'fold_cache.json',cache)
        save(OUT/f'metrics_{epoch+1:02d}.json',dict(epoch=epoch+1,train_samples=samples,mfe_feedback=feedback,
            elapsed_seconds=time.time()-started,training={k:v/samples for k,v in sums.items()},
            validation=validation_result,improved=improved))
        save(OUT/'status.json',dict(state='training',started_unix=started,heartbeat_unix=time.time(),
            elapsed_seconds=time.time()-started,completed_epochs=epoch+1,total_epochs=3,step=step,best_objective=best))
    save(OUT/'complete.json',dict(epochs=3,best_objective=best,best_sha256=digest(OUT/'best.pt'),
        original_checkpoint_unchanged=digest(CHECKPOINT)==CHECKPOINT_SHA,test_evaluated=False))
    save(OUT/'status.json',dict(state='completed',started_unix=started,heartbeat_unix=time.time(),
        elapsed_seconds=time.time()-started,completed_epochs=3,total_epochs=3,step=step,best_objective=best))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selfcheck',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        try:
            main(preflight=args.preflight)
        except BlockingIOError:
            raise
        except BaseException as error:
            if not args.preflight and (OUT/'status.json').exists():
                status=json.loads((OUT/'status.json').read_text())
                status.update(state='failed',heartbeat_unix=time.time(),error=repr(error))
                save(OUT/'status.json',status)
            raise
