import gzip,itertools,json
from collections import Counter,defaultdict
from pathlib import Path

groups=defaultdict(list)
sources=Counter()
records=0
tokens=0
for path in sorted(Path('dataset/train').glob('*.jsonl.gz')):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            records+=1
            tokens+=row['n_codons']
            sources[str(row['source'])]+=1
            groups[row['protein_sequence']].append((row['sequence_id'],row['cds_dna']))
multi={k:v for k,v in groups.items() if len(v)>1}
distances=Counter()
examples=[]
pair_count=0
for protein,items in multi.items():
    for a,b in itertools.combinations(items,2):
        pair_count+=1
        assert len(a[1])==len(b[1])
        differing=[i//3 for i in range(0,len(a[1]),3) if a[1][i:i+3]!=b[1][i:i+3]]
        distances[min(len(differing),5)]+=1
        if len(differing)==1 and len(examples)<8:
            pos=differing[0]
            masked_a=a[1][:3*pos]+'NNN'+a[1][3*pos+3:]
            masked_b=b[1][:3*pos]+'NNN'+b[1][3*pos+3:]
            assert masked_a==masked_b
            examples.append(dict(a=a[0],b=b[0],codon_position_zero_based=pos,
                                 codon_a=a[1][3*pos:3*pos+3],codon_b=b[1][3*pos:3*pos+3],
                                 identical_after_masking=True))
result=dict(train_records=records,train_tokens=tokens,sources=dict(sources),distinct_proteins=len(groups),
            proteins_with_multiple_cds=len(multi),same_protein_pairs=pair_count,
            codon_difference_histogram_5_means_at_least_5=dict(distances),examples=examples)
Path('docs/mask_plateau_data_ambiguity_20261019.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
