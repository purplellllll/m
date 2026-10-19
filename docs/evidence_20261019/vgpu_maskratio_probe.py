import json
from pathlib import Path
import torch
from latent_decoder.model import MASK,mutable_positions,synonymous_logits,CodonVMAE
from latent_decoder.train import load_data,batches
from latent_decoder.diagnose_mask_plateau import choose

torch.set_num_threads(4)
cp=torch.load('saved_weights/vmae_50ep_20261019_before_replay/last.pt',map_location='cpu')
splits,_=load_data('dataset',allow_development=True)
model=CodonVMAE(**cp['config']['model']).cuda().eval()
model.load_state_dict(cp['model'])
generator=torch.Generator(device='cuda').manual_seed(43011)
conditions=['visible','mask_005','mask_015','mask_030','eval_z_sample','train_dropout_z_sample']
totals={k:0. for k in conditions}
sites=0
with torch.no_grad():
    for batch,tokens in enumerate(batches(choose(splits['val'],128,short=True),4)):
        tokens=tokens.cuda()
        randoms=torch.rand(tokens.shape,device='cuda',generator=generator)
        mutable=mutable_positions(tokens)
        target=(randoms<.05)&mutable
        sites+=int(target.sum())
        for name in conditions:
            model.train(name=='train_dropout_z_sample')
            probability={'visible':0.,'mask_005':.05,'mask_015':.15,'mask_030':.30}.get(name,.15)
            inputs=tokens.masked_fill((randoms<probability)&mutable,MASK)
            sample=name in ('eval_z_sample','train_dropout_z_sample')
            draws=3 if sample else 1
            for draw in range(draws):
                torch.manual_seed(78211+batch*3+draw)
                logits=model(inputs,tokens,sample=sample)[0]
                nll=-synonymous_logits(logits,tokens).log_softmax(-1).gather(-1,tokens.clamp_max(63)[...,None]).squeeze(-1)
                assert torch.isfinite(nll[target]).all()
                totals[name]+=nll[target].sum().item()/draws
result=dict(validation_records=128,common_target_sites=sites,target_mask_probability=.05,stochastic_draws=3,
            ce_on_identical_targets={k:v/sites for k,v in totals.items()})
Path('runs/mask_plateau_20261019/maskratio_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
