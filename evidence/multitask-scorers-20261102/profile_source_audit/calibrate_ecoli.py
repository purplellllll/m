"""Restore observed codon profiles from author's stranded3'-end WIG files.

The author uses P-site offset14. A-site offset11 =14-3 is fixed in advance;
P-site output is retained separately rather than silently changing labels.
"""
from pathlib import Path
import sys,json,hashlib,gzip
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data.CodonTable import TranslationError
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1];WORK=RUN.parents[1]
sys.path.insert(0,str(RUN));from dynamics_v2 import fit_profile_parameters,CODON_INDEX
sys.path.insert(0,str(WORK/'.runtime/viennarna_2_7_2'));import RNA
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def structure(dna):
    n=len(dna)//3;k=min(150,len(dna));k-=k%3;s,_=RNA.fold(dna[:k].replace('T','U'));pair=np.zeros(n,dtype=int)
    pair[:k//3]=np.array([int(c!='.') for c in s]).reshape(-1,3).sum(1)
    return dict(paired=pair.tolist(),abin=[0]*n)
def metrics(rows,p,geometry):
    ys=[];ps=[];per=[]
    for r in rows:
        dna=r['dna'];n=len(dna)//3;y=np.array(r['position_rpf'],dtype=float);mask=np.isfinite(y)&(y>=0);mask[-1]=False
        y=y[mask]/y[mask].mean();ids=np.array([CODON_INDEX[dna[i:i+3]] for i in range(0,len(dna),3)])
        pos=np.arange(n);bucket=np.where(pos<n//3,0,np.where(pos<2*n//3,1,2));g=r['geometry'] if geometry else dict(paired=[0]*n,abin=[0]*n)
        pred=np.array(p['waits'])[ids]+np.array(p['pair'])[g['paired']]+np.array(p['angle'])[g['abin']]+np.array(p['bucket'])[bucket];pred=pred[mask]
        ys.extend(y);ps.extend(pred)
        if np.ptp(y)>0 and np.ptp(pred)>0:per.append(float(np.corrcoef(y,pred)[0,1]))
    y=np.array(ys);pred=np.array(ps)
    return dict(genes=len(rows),positions=len(y),pooled_relative_occupancy_pearson=float(np.corrcoef(y,pred)[0,1]),
                per_gene_pearson_mean=float(np.mean(per)),mae=float(np.mean(abs(y-pred))),
                neutral_mae=float(np.mean(abs(y-1))),warning='Direct relative-occupancy proxy diagnostic; not simulated profile or true-rate recovery')
def main():
    inputs=[HERE/x for x in ['ecoli_NC000913.2.fasta','ecoli_NC000913.2.gff3','ecoli_Mg_forward.wig','ecoli_Mg_reverse.wig']]
    save(HERE/'ecoli_calibration_protocol.json',dict(inputs={p.name:sha(p) for p in inputs},
        source='Riboformer official GitHub GSE119104 GSM3358140 high-Mg flash frozen',script_sha256=sha(Path(__file__)),
        site='Main A-site3prime_offset11=author_Psite14-3nt; Psite14 sensitivity retained',
        source_gene_split='sha256(profilecalibration20261102:proteinSHA)70/15/15 after exclusion of all task val/test exact genes,CDS,proteins',
        geometry='RNAfoldfirst150nt, codon paired bases0..3; remainder0, angle0 unobserved; same rule needed in prediction',
        labels='Observed stranded3prime-end WIG, triplet-summed; no TE or model predictions used',
        minimum='>=30sensecodons,>=20observed total reads,>=10nonzero codon positions',
        fitting='Nonnegative75-column additive relative-occupancy proxy; train-only; no true kinetics claim'))
    held=[json.loads(line) for line in (RUN/'records.jsonl').read_text(encoding='utf8').splitlines() if line]
    held=[r for r in held if r['host']=='ecoli' and r['split'] in ['val','test']]
    hp={r['protein_sha256'] for r in held};hd={hashlib.sha256(r['dna'].encode()).hexdigest() for r in held};hg={r['gene'] for r in held}
    rec=SeqIO.read(inputs[0],'fasta');genome=str(rec.seq).upper()
    assert rec.id=='NC_000913.2';counts=[np.loadtxt(p,skiprows=2,dtype=np.float64) for p in inputs[2:]]
    assert all(len(c)==len(genome) and np.isfinite(c).all() and (c>=0).all() for c in counts)
    excluded=[];accepted=[]
    for line in inputs[1].read_text().splitlines():
        if line.startswith('#'):continue
        t=line.split('\t')
        if len(t)!=9 or t[2]!='CDS':continue
        a={v.split('=',1)[0]:v.split('=',1)[1] for v in t[8].split(';') if '=' in v};start=int(t[3])-1;end=int(t[4]);strand=t[6];gene=a.get('gene',a.get('locus_tag',''))
        dna=genome[start:end] if strand=='+' else str(Seq(genome[start:end]).reverse_complement())
        if len(dna)%3 or len(dna)<93 or int(t[7])!=0:continue
        try:aa=str(Seq(dna).translate(table=11,cds=True))
        except TranslationError:continue
        ph=hashlib.sha256(aa.encode()).hexdigest();dh=hashlib.sha256(dna.encode()).hexdigest()
        if gene in hg or a.get('locus_tag') in hg or ph in hp or dh in hd:
            excluded.append(dict(gene=gene,locus=a.get('locus_tag'),reason='task_val_test_gene_protein_or_DNA_overlap'));continue
        if start<14 or end+14>len(genome):continue
        ys={}
        for label,offset in [('Asite11',11),('Psite14',14)]:
            nt=counts[0][start+offset:end+offset] if strand=='+' else counts[1][start-offset:end-offset][::-1]
            assert len(nt)==len(dna);ys[label]=nt.reshape(-1,3).sum(1)
        y=ys['Asite11'];
        if y[:-1].sum()<20 or np.count_nonzero(y[:-1])<10:continue
        rank=int(hashlib.sha256(('profilecalibration20261102:'+ph).encode()).hexdigest()[:8],16)/2**32
        split='train' if rank<.7 else 'val' if rank<.85 else 'test'
        accepted.append(dict(split=split,host='ecoli',gene=gene,locus_tag=a.get('locus_tag'),dna=dna,protein_sha256=ph,geometry=structure(dna),
            position_rpf=ys['Asite11'].tolist(),Psite14=ys['Psite14'].tolist(),source_coordinate=dict(start0=start,end0=end,strand=strand)))
    assert len(accepted)>100
    assert not {r['protein_sha256'] for r in accepted if r['split']=='train'}&{r['protein_sha256'] for r in accepted if r['split']!='train'}
    # Check exact strand-aware source formula independently on stored profiles.
    for r in accepted[:8]:
        p=r['source_coordinate'];start=p['start0'];end=p['end0'];strand=p['strand'];arr=counts[0] if strand=='+' else counts[1]
        coordinates=range(start,end) if strand=='+' else range(end-1,start-1,-1)
        vals=[arr[x+11 if strand=='+' else x-11] for x in coordinates]
        np.testing.assert_array_equal(np.array(vals).reshape(-1,3).sum(1),r['position_rpf'])
    train=[r for r in accepted if r['split']=='train'];val=[r for r in accepted if r['split']=='val'];test=[r for r in accepted if r['split']=='test'];models=[]
    for site,geom in [('Asite11',False),('Asite11',True),('Psite14',True)]:
        fitrows=[]
        for r in train:
            q=dict(r)
            if not geom:q.pop('geometry')
            if site=='Psite14':q['position_rpf']=q['Psite14']
            fitrows.append(q)
        params=fit_profile_parameters(fitrows,'ecoli','Riboformer_GSE119104_GSM3358140_'+site+('_5prime150' if geom else '_codon'))
        params.update(site_assignment=site,position_count_definition='triplet-summed observed3prime-end WIG with documented strand-aware offset',
            geometry_rule='RNAfold_first150nt_codon_paired_base_count_rest0_angle0' if geom else 'none',
            source_input_sha256={p.name:sha(p) for p in inputs},current_task_heldout_excluded=True)
        name='ecoli_'+site+('_5prime150' if geom else '_codon')+'_parameters.json';save(HERE/name,params)
        vv=[];tt=[]
        for dest,source in [(vv,val),(tt,test)]:
            for r in source:
                q=dict(r)
                if site=='Psite14':q['position_rpf']=r['Psite14']
                dest.append(q)
        models.append(dict(model=name,sha256=sha(HERE/name),validation=metrics(vv,params,geom),test=metrics(tt,params,geom)))
    with gzip.open(HERE/'ecoli_observed_profiles.jsonl.gz','wt',encoding='utf8') as f:
        for r in accepted:f.write(json.dumps(r,allow_nan=False)+'\n')
    save(HERE/'ecoli_calibration_summary.json',dict(status='completed',reference=rec.id,profile_source='Observed RPF, no synthetic supervision',
        counts={s:sum(r['split']==s for r in accepted) for s in ['train','val','test']},excluded_task_val_test=len(excluded),exclusions=excluded,
        source_formula_verified_examples=8,models=models,
        limits=['P-site to A-site one-codon conversion predeclared, not learned from heldout',
                'Relative occupancy proxy not identifiable physical waiting rate; initiation not calibrated',
                'Exact protein/gene/CDS isolation only; no near-homology guarantee']))
    print(json.dumps(dict(counts={s:sum(r['split']==s for r in accepted) for s in ['train','val','test']},excluded=len(excluded),models=models),indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
