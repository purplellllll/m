"""Isolated single-record replay; never writes to the source run/checkpoint."""
import argparse
import copy
import json
import math
from pathlib import Path

import torch

from .model import CodonVMAE, collate_records, corrupt, synonymous_logits, vae_loss
from .train import load_data, sha256


def gradient_groups(model):
    groups = {}
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Nonfinite gradient: ' + name)
        groups.setdefault(name.split('.')[0], []).append(parameter.grad.float().square().sum())
    return {k: torch.stack(v).sum().sqrt().item() for k, v in groups.items()}, missing


def changes(model, before):
    groups = {}
    for name, parameter in model.named_parameters():
        delta = parameter.detach() - before[name]
        values = torch.stack(((delta != 0).sum().float(), delta.square().sum(),
                              before[name].square().sum()))
        groups.setdefault(name.split('.')[0], []).append(values)
    result = {}
    for name, rows in groups.items():
        changed, delta_sq, weight_sq = torch.stack(rows).sum(0).tolist()
        result[name] = dict(changed_elements=int(changed), delta_l2=math.sqrt(delta_sq),
                            relative_delta_l2=math.sqrt(delta_sq / max(weight_sq, 1e-30)))
    return result


def objective(model, tokens, corrupted, selected, training, beta=None):
    logits, mu, logvar = model(corrupted, tokens)
    loss, terms = vae_loss(logits, tokens, mu, logvar, selected,
                          training['beta_kl'] if beta is None else beta,
                          training['masked_loss_weight'], training.get('full_loss_weight', 1.))
    return loss, terms, logits


@torch.no_grad()
def probe(model, tokens, corrupted, selected, training):
    was_training = model.training
    model.eval()
    logits, mu, logvar = model(corrupted, tokens, sample=False)
    loss, terms = vae_loss(logits, tokens, mu, logvar, selected, training['beta_kl'],
                          training['masked_loss_weight'], training.get('full_loss_weight', 1.))
    accuracy = (synonymous_logits(logits, tokens).argmax(-1)[selected] == tokens[selected]).float().mean()
    model.train(was_training)
    return dict(loss=loss.item(), masked_ce=terms['masked_ce'].item(), masked_accuracy=accuracy.item())


def run_arm(checkpoint, tokens, corrupted, selected, *, lr, steps, dynamic=False, log=None):
    config = checkpoint['config']
    training = config['training']
    model = CodonVMAE(**config['model']).to(tokens.device)
    model.load_state_dict(checkpoint['model'])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                  weight_decay=training['weight_decay'])
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    for group in optimizer.param_groups:
        group['lr'] = lr
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
    assert all(p.requires_grad for p in model.parameters())
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    initial_probe = probe(model, tokens, corrupted, selected, training)
    initial_counts = [float(s['step']) for s in optimizer.state.values()]
    accum = training['gradient_accumulation']
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        total = 0.
        first_objective = None
        for micro in range(accum):
            seed = 91273 + micro + (step * accum if dynamic else 0)
            torch.manual_seed(seed)
            if dynamic:
                inputs, mask = corrupt(tokens, training['mask_probability'])
            else:
                inputs, mask = corrupted, selected
            loss, terms, _ = objective(model, tokens, inputs, mask, training)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite replay loss')
            (loss / accum).backward()
            total += loss.item() / accum
            if micro == 0:
                first_objective = loss.item()
        detailed = step in (1, steps)
        grad_groups, missing = gradient_groups(model) if detailed else ({}, [])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        delta = changes(model, before)
        assert norm.item() > 0
        count = sum(v['changed_elements'] for v in delta.values())
        assert (count == 0) if lr == 0 else (count > 0)
        row = dict(update=step, lr=lr, train_loss=total, grad_norm=norm.item(),
                   changed_elements=count, update_delta_l2=math.sqrt(sum(v['delta_l2']**2 for v in delta.values())))
        if detailed:
            row.update(parameter_groups=delta, gradient_groups=grad_groups, missing_gradients=missing)
        if step == 1 or step % 10 == 0 or step == steps:
            row['fixed_eval_probe'] = probe(model, tokens, corrupted, selected, training)
            if not dynamic:
                with torch.no_grad():
                    torch.manual_seed(91273)
                    after = objective(model, tokens, corrupted, selected, training)[0].item()
                row.update(first_micro_loss_before=first_objective, first_micro_loss_after=after)
            print(json.dumps(dict(event='replay_update', dynamic=dynamic, **row)), flush=True)
        history.append(row)
        if log is not None:
            with log.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
    final_counts = [float(s['step']) for s in optimizer.state.values()]
    assert all(b-a == steps for a,b in zip(initial_counts, final_counts))
    final = dict(lr=lr, dynamic=dynamic, updates=steps, initial_probe=initial_probe,
                 final_probe=probe(model, tokens, corrupted, selected, training),
                 initial_optimizer_step_range=[min(initial_counts), max(initial_counts)],
                 final_optimizer_step_range=[min(final_counts), max(final_counts)],
                 total_parameter_changes=changes(model, initial),
                 first_train_loss=history[0]['train_loss'], final_train_loss=history[-1]['train_loss'])
    if log is not None:
        torch.save(dict(model=model.state_dict(), source_epoch=checkpoint['next_epoch'],
                        diagnostic_only=True, lr=lr, updates=steps, config=config),
                   log.with_suffix('.model.pt'))
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=60)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError('steps must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_path = Path(args.checkpoint)
    digest = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    splits, _ = load_data(args.data, allow_development=True)
    eligible = [r for r in splits['train'] if 256 <= r['n_codons'] <= 320]
    record = sorted(eligible, key=lambda r: sha256_text(r['cds_dna']))[0]
    (out/'single_record.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training = checkpoint['config']['training']
    tokens = collate_records([record] * training['batch_size']).to(device)
    torch.manual_seed(61203)
    corrupted, selected = corrupt(tokens[:1], training['mask_probability'])
    corrupted = corrupted.expand_as(tokens).clone()
    selected = selected.expand_as(tokens).clone()
    assert int(selected[0].sum()) >= 10
    torch.save(dict(tokens=tokens.cpu(), corrupted=corrupted.cpu(), selected=selected.cpu()), out/'fixed_batch.pt')
    model = CodonVMAE(**checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.train()
    torch.manual_seed(91273)
    reconstruction = objective(model,tokens,corrupted,selected,training,beta=0.)[0]
    reconstruction.backward()
    groups, missing = gradient_groups(model)
    assert groups['encoder'] > 0 and groups['mu'] > 0 and groups['decoder'] > 0
    audit = dict(reconstruction_only_gradient_groups=groups, missing_gradients=missing,
                 checkpoint_epoch=checkpoint['next_epoch'], checkpoint_lr=checkpoint['optimizer']['param_groups'][0]['lr'],
                 codons=record['n_codons'], selected_sites_per_record=int(selected[0].sum()),
                 batch_size=training['batch_size'], gradient_accumulation=training['gradient_accumulation'])
    del model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    print(json.dumps(dict(event='gradient_audit', **audit)), flush=True)
    results = {}
    for name, lr, steps, dynamic in (
            ('zero_lr', 0., 3, False), ('tail_lr', 7.596123493895991e-7, args.steps, False),
            ('reset_lr', 1e-4, args.steps, False), ('random_mask_lr', 1e-4, args.steps, True)):
        results[name] = run_arm(checkpoint,tokens,corrupted,selected,lr=lr,steps=steps,dynamic=dynamic,log=out/(name+'.jsonl'))
        (out/'summary.json').write_text(json.dumps(dict(audit=audit,arms=results),indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(event='arm_complete',name=name,**results[name])),flush=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    assert sha256(checkpoint_path) == digest
    (out/'source_sha256_verified.txt').write_text(digest+'\n')
    print(json.dumps(dict(event='replay_complete', source_unchanged=True, output=str(out))),flush=True)


def sha256_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__':
    main()
