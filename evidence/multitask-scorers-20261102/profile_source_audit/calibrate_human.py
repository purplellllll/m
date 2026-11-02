"""Observed human5prime footprint-start profiles; offsets are explicit hypotheses."""
from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np
import h5py
from Bio import SeqIO
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1];WORK=RUN.parents[1]
sys.path.insert(0,str(HERE));from calibrate_ecoli import structure,metrics,save,sha
sys.path.insert(0,str(RUN));from dynamics_v2 import fit_profile_parameters
def main():
    files=[HERE/n for n in ['human_GSM1606107_coverage.ribo','human_APPRIS_transcripts.fa.gz','official_coverage.py','official_get_transcript_coverage.py']]
    save(HERE/'human_calibration_protocol.json',dict(source='Official ribosomeprofiling/ribo_manuscript_supplemental GSM1606107 HEK293 GSE65778; APPRIS2019 reference',
        input_sha256={p.name:sha(p) for p in files},script_sha256=sha(Path(__file__)),
        position_definition='Official ribopy coverage stores aligned read5prime STARTS, not P/A-site and not span-coverage',
        reads='Fixed28..30nt read lengths only',site='Predeclared A-site hypothesis start+15nt; Psite start+12nt sensitivity; not experimentally calibrated offsets',CDS='Official annotation end excludes terminal stop; append next3 reference bases and require canonical stop',
        split='Source gene hash70/15/15 after exclusion all currenthuman task val/test gene,CDS,protein, plus oldhumanTE val/test genes',
        geometry='RNAfoldfirst150nt codon paired counts;rest0;angle0',
        fitting='Train-only relative-occupancy proxy, no true kinetics recovery; no heldout site-offset tuning',
        filters='Complete ATG-CDS with single terminalstop,>=90nt,<=4500nt,>=20reads/10nonzero positions'))
    data=[json.loads(l) for l in (RUN/'records.jsonl').read_text(encoding='utf8').splitlines()]
    held=[r for r in data if r['host']=='human' and r['split']!='train'];hg={r['gene'].split('.')[0] for r in held};hp={r['protein_sha256'] for r in held};hd={r['uid'] for r in held}
    old=json.loads((WORK/'runs/human_te_expanded_20261101/train_validation.json').read_text(encoding='utf8'))
    hg|={r['gene'].split('.')[0] for r in old if r['split']=='val'}
    hg|={r['gene'].split('.')[0] for r in json.loads((WORK/'runs/human_te_expanded_20261101/test_inputs.json').read_text(encoding='utf8'))}
    seqs={r.id:str(r.seq).upper() for r in SeqIO.parse(gzip.open(files[1],'rt'),'fasta')}
    records=[];exclusions=[]
    with h5py.File(files[0],'r') as f:
        names=[x.decode() for x in f['reference/reference_names'][:]];lens=f['reference/reference_lengths'][:].astype(np.int64);annot=f['reference/annotation'][:];total=int(lens.sum());offsets=np.r_[0,np.cumsum(lens)]
        coverage=f['experiments/GSM1606107/coverage/coverage'];assert coverage.shape==(total*26,)
        # Sum only three prespecified read lengths; preserve observed5prime coordinates.
        nt=np.zeros(total,dtype=np.uint32)
        for length in [28,29,30]:
            nt+=coverage[(length-15)*total:(length-14)*total].astype(np.uint32)
        for i,name in enumerate(names):
            if name not in seqs:continue
            sequence=seqs[name];assert len(sequence)==int(lens[i]);start,end,_=map(int,annot[i]);end+=3;dna=sequence[start:end]
            if len(dna)<90 or len(dna)>4500 or len(dna)%3 or not dna.startswith('ATG') or dna[-3:] not in {'TAA','TAG','TGA'} or set(dna)-set('ACGT'):continue
            cods=[dna[j:j+3] for j in range(0,len(dna),3)]
            if any(c in {'TAA','TAG','TGA'} for c in cods[:-1]):continue
            from Bio.Seq import Seq
            protein=str(Seq(dna[:-3]).translate());gene=name.split('|')[1].split('.')[0];ph=hashlib.sha256(protein.encode()).hexdigest();dh=hashlib.sha256(dna.encode()).hexdigest()
            if gene in hg or ph in hp or dh in hd:exclusions.append(dict(gene=gene,reason='task_or_oldTE_heldout_overlap'));continue
            vector=nt[offsets[i]:offsets[i+1]];ys={}
            for site,shift in [('Asite15',15),('Psite12',12)]:
                positions=np.arange(start,end)-shift;valid=(positions>=0)&(positions<len(vector));v=np.zeros(len(dna),dtype=float);v[valid]=vector[positions[valid]];ys[site]=v.reshape(-1,3).sum(1)
            if ys['Asite15'][:-1].sum()<20 or np.count_nonzero(ys['Asite15'][:-1])<10:continue
            rank=int(hashlib.sha256(('profilecalibration20261102:'+gene).encode()).hexdigest()[:8],16)/2**32
            records.append(dict(host='human',split='train' if rank<.7 else 'val' if rank<.85 else 'test',gene=gene,transcript=name,dna=dna,protein_sha256=ph,position_rpf=ys['Asite15'].tolist(),Psite12=ys['Psite12'].tolist(),geometry=structure(dna)))
    # One source transcript per gene; source APPRIS generally already one.
    unique={}
    for r in records:
        if r['gene'] not in unique:unique[r['gene']]=r
    records=list(unique.values());assert len(records)>100
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val'];test=[r for r in records if r['split']=='test']
    assert not {r['gene'] for r in train}&({r['gene'] for r in val+test}|hg)
    models=[]
    for site in ['Asite15','Psite12']:
        fit=[]
        for r in train:
            q=dict(r)
            if site=='Psite12':q['position_rpf']=r['Psite12']
            fit.append(q)
        params=fit_profile_parameters(fit,'human','HEK293_GSE65778_GSM1606107_5primeStarts28to30nt_'+site+'_5prime150')
        params.update(geometry_rule='RNAfold_first150nt_codon_paired_base_count_rest0_angle0',site_assignment=site,
            source_input_sha256={p.name:sha(p) for p in files},current_task_heldout_excluded=True,offset_experimentally_calibrated=False)
        path=HERE/('human_'+site+'_5prime150_parameters.json');save(path,params)
        groups=[]
        for subset in [val,test]:
            target=[]
            for r in subset:
                q=dict(r)
                if site=='Psite12':q['position_rpf']=r['Psite12']
                target.append(q)
            groups.append(metrics(target,params,True))
        models.append(dict(model=path.name,sha256=sha(path),validation=groups[0],test=groups[1]))
    with gzip.open(HERE/'human_observed_profiles.jsonl.gz','wt',encoding='utf8') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    save(HERE/'human_calibration_summary.json',dict(status='completed',counts={s:sum(r['split']==s for r in records) for s in ['train','val','test']},excluded_task_heldout=len(exclusions),models=models,
        limits=['5prime footprint starts shifted with prespecified offsets, not experimentally inferred P/A sites','Relative RPF proxy not identified kinetic parameters','One human experiment, no nearhomology split guarantee']))
    print(json.dumps(dict(counts={s:sum(r['split']==s for r in records) for s in ['train','val','test']},models=models),indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
