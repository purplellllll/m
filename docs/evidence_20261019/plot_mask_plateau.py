import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('tmp/mask_plateau_collected_20261019/runs/mask_plateau_20261019')
summary=json.loads((root/'summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
for name,label,color in [('stream_tail','LR 7.60e-7, beta .001','#777777'),('stream_reset','LR 1e-4, beta .001','#1976b5'),('stream_no_kl','LR 1e-4, beta 0','#e89027')]:
    points=summary['arms'][name]['points']
    axes[0].plot([p['step'] for p in points],[p['probes']['val']['masked_ce'] for p in points],marker='o',ms=3,label=label,color=color)
axes[0].set_ylim(.85,1.15)
axes[0].set_title('Broad training stream: held-out probe')
axes[0].set_ylabel('Masked CE on 128 validation CDS')
axes[0].legend(fontsize=8)
for name,probe,label,color,style in [
    ('subset_fixed','train_mask_A','Fixed mask: trained positions','#1976b5','-'),
    ('subset_fixed','train_mask_B','Fixed mask: new mask, same CDS','#e89027','--'),
    ('subset_fixed','val','Fixed mask: validation CDS','#ba4040','-'),
    ('subset_dynamic','train_mask_A','Dynamic mask: training CDS','#209b75','-'),
    ('subset_dynamic','val','Dynamic mask: validation CDS','#8c63ad','--')]:
    points=summary['arms'][name]['points']
    axes[1].plot([p['step'] for p in points],[p['probes'][probe]['masked_ce'] for p in points],label=label,color=color,ls=style,marker='o',ms=3)
axes[1].set_yscale('log')
axes[1].set_title('Repeated 128-CDS subset: fit vs transfer')
axes[1].set_ylabel('Masked CE (log scale)')
axes[1].legend(fontsize=7.5,loc='lower left')
family=summary['audit']['val']['family']
sizes=[2,3,4,6]
x=np.arange(len(sizes))
axes[2].bar(x-.18,[family[str(k)]['ce'] for k in sizes],.36,label='Model',color='#1976b5')
axes[2].bar(x+.18,[math.log(k) for k in sizes],.36,label='Uniform among synonyms',color='#c4c8cd')
axes[2].set_xticks(x,[str(k) for k in sizes])
axes[2].set_xlabel('Number of synonymous codons')
axes[2].set_ylabel('Masked CE')
axes[2].set_title('Error grows with synonymous choices')
axes[2].legend(fontsize=8)
for ax in axes[:2]:
    ax.set_xlabel('Optimizer update')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
fig.suptitle('Masked-loss plateau diagnosis | original 50-epoch checkpoint | exploratory controls',fontsize=13)
fig.savefig('docs/mask_plateau_diagnosis_20261019.png',dpi=170)
fig.savefig('docs/mask_plateau_diagnosis_20261019.svg')
print('Saved diagnostic plots')
