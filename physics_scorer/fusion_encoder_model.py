"""Three-way fusion gains and small codon-level context architecture controls."""
import torch
from torch import nn
from .profile_model import AlignedProfileModel


class HuggingFaceContext(nn.Module):
    def __init__(self,kind,width,ff,embedding):
        super().__init__()
        if kind=='bert':
            from transformers import BertConfig,BertModel
            config=BertConfig(vocab_size=65,hidden_size=width,num_hidden_layers=1,num_attention_heads=8,
                intermediate_size=ff,max_position_embeddings=256,type_vocab_size=1,pad_token_id=64,
                hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,use_cache=False)
            config._attn_implementation='eager'
            self.encoder=BertModel(config,add_pooling_layer=False)
        elif kind=='t5':
            from transformers import T5Config,T5EncoderModel
            config=T5Config(vocab_size=65,d_model=width,d_kv=width//8,d_ff=ff,num_layers=1,
                num_heads=8,dropout_rate=.1,feed_forward_proj='relu',pad_token_id=64,use_cache=False)
            config._attn_implementation='eager'
            self.encoder=T5EncoderModel(config)
        else:raise ValueError('Unknown context family')
        self.encoder.set_input_embeddings(embedding)

    def forward(self,x,src_key_padding_mask):
        return self.encoder(inputs_embeds=x,attention_mask=(~src_key_padding_mask).long(),return_dict=True).last_hidden_state


class FusionEncoderModel(AlignedProfileModel):
    def __init__(self,encoder='bilstm',width=424,gains=(1.,1.,1.),learned_gains=False):
        if encoder not in {'bilstm','bilstm_ln','transformer','bert','t5'}:raise ValueError('Unknown encoder')
        # Initialize shared branches and fusion identically for a given seed.
        super().__init__(encoder='bilstm',width=width)
        if len(gains)!=3 or min(gains)<0:raise ValueError('Three nonnegative gains required')
        self.family=encoder;self.width=width;self.learned_gains=learned_gains
        self.register_buffer('fixed_gains',torch.tensor(gains,dtype=torch.float32))
        self.gain_logits=nn.Parameter(torch.zeros(3),requires_grad=learned_gains)
        self.context_output_norm=nn.LayerNorm(width) if encoder=='bilstm_ln' else nn.Identity()
        self._codon_cache=None;self._active=False;self._gain_override=None
        target=sum(p.numel() for p in self.parameters())
        self.context_ff=None
        if encoder in {'transformer','bert','t5'}:
            self.encoder_kind='transformer'
            self.context=self._context(encoder,width,width)
            count=sum(p.numel() for p in self.parameters());slope=2*width+(0 if encoder=='t5' else 1)
            ff=max(8,int(round((width+(target-count)/slope)/8))*8)
            self.context=self._context(encoder,width,ff);self.context_ff=ff
            if encoder in {'bert','t5'}:
                self.codons.register_forward_hook(self._cache_codons)
                self.context.register_forward_pre_hook(self._replace_positional_input,with_kwargs=True)
        self.fusion[0].register_forward_pre_hook(self._scale_fusion)

    def _context(self,kind,width,ff):
        if kind=='transformer':
            layer=nn.TransformerEncoderLayer(width,8,ff,.1,'gelu',batch_first=True,norm_first=True)
            return nn.TransformerEncoder(layer,1,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        return HuggingFaceContext(kind,width,ff,self.codons)

    def _cache_codons(self,module,args,out):
        if self._active:self._codon_cache=out

    def _replace_positional_input(self,module,args,kwargs):
        # BERT supplies learned absolute positions; T5 supplies relative attention
        # bias. Pass raw codon embeddings rather than adding sinusoidal positions.
        return (self._codon_cache,*args[1:]),kwargs

    def effective_gains(self):
        if self._gain_override is not None:return self._gain_override
        return self.fixed_gains*(3*self.gain_logits.softmax(0) if self.learned_gains else 1)

    def _scale_fusion(self,module,args):
        x=args[0];p,g,s=x.split(self.width,dim=-1)
        s=self.context_output_norm(s)
        gains=self.effective_gains()
        return (torch.cat((p*gains[0],g*gains[1],s*gains[2]),-1),*args[1:])

    def forward(self,*args,gain_override=None,**kwargs):
        try:
            self._active=True
            if gain_override is not None:
                gains=torch.as_tensor(gain_override,device=args[0].device,dtype=self.fixed_gains.dtype)
                if gains.shape!=(3,) or not torch.isfinite(gains).all() or (gains<0).any():raise ValueError('Invalid gain override')
                self._gain_override=gains
            return super().forward(*args,**kwargs)
        finally:self._active=False;self._codon_cache=None;self._gain_override=None
