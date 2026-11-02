"""Conditional latent denoising diffusion; not a discrete-CDS generator or GNN."""
import torch
from torch import nn
from torch.nn import functional as F
from base_model import MultiTaskTransformer as Base,TASKS,masked_loss
class MultiTaskTransformer(Base):
    def __init__(self,stats_dim,width=192,use_physics=True,max_tokens=512):
        super().__init__(stats_dim,width,use_physics,max_tokens)
        state=torch.get_rng_state()
        self.time_embedding=nn.Embedding(16,96)
        self.noise_input=nn.Conv1d(2*width,96,3,padding=1)
        self.noise_middle=nn.Conv1d(96,96,3,padding=1)
        self.noise_output=nn.Conv1d(96,width,1)
        self.diffusion_gate=nn.Parameter(torch.zeros(()))
        self.register_buffer('alpha_bar',torch.cumprod(1-torch.linspace(.01,.25,16),0))
        self.register_buffer('diffusion_noise_counter',torch.zeros((),dtype=torch.long))
        torch.set_rng_state(state) # Preserve common-backbone/dropout RNG pairing.
        self.diffusion_loss=torch.tensor(0.);self._noise_cache={}
        self.context.register_forward_hook(self._diffuse,with_kwargs=True)
    def _epsilon(self,z,condition,t,valid):
        z=z.masked_fill(~valid[...,None],0);condition=condition.masked_fill(~valid[...,None],0)
        h=self.noise_input(torch.cat([z,condition],-1).transpose(1,2))+self.time_embedding(t)[:,:,None]
        h=F.gelu(h).masked_fill(~valid[:,None],0)
        h=F.gelu(self.noise_middle(h)).masked_fill(~valid[:,None],0)
        return self.noise_output(h).transpose(1,2).masked_fill(~valid[...,None],0)
    def _diffuse(self,module,args,kwargs,output):
        condition=args[0];valid=~kwargs['src_key_padding_mask'];b,n,w=output.shape
        if self.training:
            devices=[output.device.index] if output.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(20261102+int(self.diffusion_noise_counter.item()))
                t=torch.randint(0,16,(b,),device=output.device);eps=torch.randn_like(output)
            self.diffusion_noise_counter.add_(1);a=self.alpha_bar[t][:,None,None]
            noisy=a.sqrt()*output.detach()+(1-a).sqrt()*eps
            estimate=self._epsilon(noisy,condition,t,valid)
            self.diffusion_loss=(((estimate-eps).square())*valid[...,None]).sum()/(valid.sum()*w)
            clean=((noisy-(1-a).sqrt()*estimate)/a.sqrt()).clamp(-5,5)
        else:
            key=(n,str(output.device))
            if key not in self._noise_cache:
                rng=torch.Generator(device='cpu').manual_seed(20261102)
                self._noise_cache[key]=torch.randn(1,n,w,generator=rng).to(output.device)
            z=self._noise_cache[key].expand(b,-1,-1).masked_fill(~valid[...,None],0)
            times=[15,10,5,0]
            for k,time in enumerate(times):
                t=torch.full((b,),time,device=output.device,dtype=torch.long);a=self.alpha_bar[time]
                estimate=self._epsilon(z,condition,t,valid);clean=((z-(1-a).sqrt()*estimate)/a.sqrt()).clamp(-5,5)
                previous=self.alpha_bar[times[k+1]] if k+1<len(times) else a.new_tensor(1.)
                z=(previous.sqrt()*clean+(1-previous).sqrt()*estimate).masked_fill(~valid[...,None],0)
            self.diffusion_loss=output.new_zeros(())
        return (output+.1*torch.tanh(self.diffusion_gate)*(clean-output)).masked_fill(~valid[...,None],0)
def selfcheck():
    from base_model import selfcheck as base_check
    base_check();torch.manual_seed(30101)
    base=Base(285);torch.manual_seed(30101);m=MultiTaskTransformer(285)
    for k,v in base.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
    b,n=2,20
    x=dict(codons=torch.randint(0,64,(b,n)),amino=torch.randint(0,20,(b,n)),valid=torch.ones(b,n,dtype=torch.bool),physics=torch.randn(b,n,12),physics_present=torch.ones(b),physical_global=torch.randn(b,36),physics_calibrated=torch.ones(b),utr=torch.randint(0,4,(b,2,30)),utr_present=torch.ones(b,2),stats=torch.randn(b,285),host=torch.tensor([0,1]))
    m.train();p=m(x);target=torch.randn_like(p);mask=torch.ones_like(p,dtype=torch.bool);loss=masked_loss(p,target,mask)+.05*m.diffusion_loss;loss.backward()
    assert m.noise_output.weight.grad.abs().sum()>0 and m.context.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    m.eval();m.diffusion_gate.data.fill_(.3);a=m(x);z=m(x);torch.testing.assert_close(a,z,atol=0,rtol=0)
    print('PASS identical common initialization, denoising gradient and deterministic DDIM')
if __name__=='__main__':selfcheck()
