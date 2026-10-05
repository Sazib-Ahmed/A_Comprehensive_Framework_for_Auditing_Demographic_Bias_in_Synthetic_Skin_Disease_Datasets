"""Reconstruct the historical image-level splitter from the original ordered rows.

This is not a historical manifest and does not remove source/patient leakage.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from audit_common import new_output,write_json,sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata',required=True,help='02 audit_metadata.csv in unchanged original row order')
    ap.add_argument('--confirm-original-row-order',action='store_true');ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if not args.confirm_original_row_order:raise ValueError('Original ordered records required')
    d=pd.read_csv(args.metadata,dtype={'sample_id':str})
    if d.sample_id.duplicated().any() or d[['sample_id','label']].isna().any().any():raise ValueError('Invalid metadata')
    y=d.label.astype(str).to_numpy();outer=StratifiedKFold(n_splits=7,shuffle=True,random_state=42)
    rows=[]
    for f,(rest,te) in enumerate(outer.split(np.arange(len(d)),y),1):
        inner=StratifiedShuffleSplit(n_splits=1,test_size=.175,random_state=42+f)
        tr_rel,va_rel=next(inner.split(rest,y[rest]))
        b=d[['sample_id','label']].copy();b['fold']=f;b['role']=''
        b.loc[rest[tr_rel],'role']='Train';b.loc[rest[va_rel],'role']='Val';b.loc[te,'role']='Test';rows.append(b)
    out=new_output(args.out);m=pd.concat(rows,ignore_index=True)
    m.to_csv(out/'reconstructed_fold_manifest.csv',index=False)
    m.groupby(['fold','role','label']).size().rename('n').reset_index().to_csv(out/'fold_counts.csv',index=False)
    write_json(out/'provenance.json',dict(metadata_sha256=sha256(args.metadata),split_seed=42,
        validation_seed='42+fold',validation_fraction_of_remainder=.175,
        limitation='Reconstruction, not historically saved identity evidence. Old CSV order must be confirmed. No patient/duplicate grouping.'))


if __name__=='__main__':main()
