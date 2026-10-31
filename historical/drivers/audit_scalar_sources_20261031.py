"""Audit task schemas before creating human scalar targets or model splits."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
OUT=WORK/'runs/scalar_downstream_20261031'


def main():
    OUT.mkdir(exist_ok=True)
    path=WORK/'data/official/dataset_for_translation_model.zip'
    assert hashlib.md5(path.read_bytes()).hexdigest()=='9e73fdb5b25aa063a8152b541ac302d5'
    with zipfile.ZipFile(path) as z:
        genes=pd.read_csv(z.open('genes.csv'));meta=pd.read_csv(z.open('sample_information.csv'))
        rna=pd.read_csv(z.open('RNA_RPKM_320.csv'),index_col=0)
        rpf=pd.read_csv(z.open('RPF_RPKM_320.csv'),index_col=0)
        readme=z.read('README.txt').decode(errors='replace')
    table=WORK/'deliverables/public_riboseq_validation_20261028/human_TE_Supplementary_Table1.xlsx'
    te=pd.read_excel(table)
    result=dict(ribodecode=dict(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n_train_genes=int(genes.Training_set.notna().sum()),n_test_genes=int(genes.Test_set.notna().sum()),
        n_environments=len(meta),metadata_columns=meta.columns.tolist(),
        environment_types={str(k):int(v) for k,v in meta.Type.value_counts().items()},
        expression_shapes=[list(rna.shape),list(rpf.shape)],readme=readme),
        seq2ribo_te=dict(source_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),n_genes=len(te),
        verified_label_mapping={'mean':'mean_te','hek293':'TE_HEK293','lcl':'TE_LCL','rpe':'TE_RPE-1'},
        finite_labels={c:int(np.isfinite(te[c]).sum()) for c in ['mean_te','TE_HEK293','TE_LCL','TE_RPE-1']},
        provided_fold_counts={str(k):int(v) for k,v in te.fold.value_counts().items()},
        fold_identity='RiboNN folds; NOT automatically seq2ribo native splits'))
    (OUT/'source_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
