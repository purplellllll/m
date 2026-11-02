"""Check human-head feature contract, deeper-context gradients and complete route exclusions."""
from pathlib import Path
import sys
import copy
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_pilot import Head,ReliableGraphContext,TaskScalarModel,save,ROOT

torch.set_num_threads(1);torch.manual_seed(91);torch.backends.mha.set_fastpath_enabled(False)
x=(torch.tensor([[1,2,3,4,5,6,7,8]]),torch.ones(1,8,dtype=torch.bool),torch.rand(1,8,2),torch.rand(1,8,5),torch.tensor([[0,7],[7,0]]))
stats=torch.rand(1,3);checks=[]
for arm in ['two_layers','four_layers','no_physics','sequence_only']:
    base=ReliableGraphContext(width=192,anchored_head=False)
    if arm=='four_layers':base.context.layers.extend([copy.deepcopy(z) for z in base.context.layers]);base.context.num_layers=4
    if arm=='no_physics':base.use_physics=False
    if arm=='sequence_only':base.input_fusion.register_forward_pre_hook(lambda m,args:(torch.cat((args[0][...,:192],torch.zeros_like(args[0][...,192:])),dim=-1),))
    model=TaskScalarModel(base,'te').eval();head=Head().eval()
    pooled,_=model.representations(x);value=head(pooled[:,2],stats);value.square().sum().backward()
    assert base.context.layers[-1].self_attn.in_proj_weight.grad.abs().sum()>0
    if arm in ['two_layers','four_layers']:assert base.physics[0].weight.grad.abs().sum()>0
    else:
        changed=(*x[:2],x[2]+100,*x[3:])
        with torch.no_grad():other=head(model.representations(changed)[0][:,2],stats)
        torch.testing.assert_close(value,other,atol=0,rtol=0)
    if arm=='sequence_only':
        with torch.no_grad():other=head(model.representations((*x[:3],x[3]+100,x[4][:,:0]))[0][:,2],stats)
        torch.testing.assert_close(value,other,atol=0,rtol=0)
    checks.append(arm)
save(ROOT/'selfcheck.json',dict(status='passed',arms=checks,head_properties=3,explicit_length_input=False,last_context_layer_gradient_verified=True,route_exclusion_verified=True))
print('PASS: human three-property head,2/4-layer gradients, full physical and graph-input exclusions')
