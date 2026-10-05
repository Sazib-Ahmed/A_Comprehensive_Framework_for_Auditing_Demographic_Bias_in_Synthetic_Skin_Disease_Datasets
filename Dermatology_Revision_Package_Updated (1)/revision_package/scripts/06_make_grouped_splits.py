"""Create NEW leakage-aware folds from verified identity links. Never changes old folds."""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from audit_common import new_output,write_json,sha256


class UnionFind:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata',required=True,help='Audit metadata with sample_id,label and optional patient_id,case_id,parent_id,acquisition_session')
    ap.add_argument('--reviewed-pairs',help='02 candidates after review: sample_id_a,sample_id_b,verified_related=yes/no')
    ap.add_argument('--folds',type=int,default=7);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--out',required=True)
    args=ap.parse_args();d=pd.read_csv(args.metadata,dtype={'sample_id':str})
    if d.sample_id.duplicated().any() or d[['sample_id','label']].isna().any().any():raise ValueError('Invalid IDs/labels')
    uf=UnionFind()
    for sid in d.sample_id:uf.find(sid)
    linkage=[]
    for key in ['patient_id','case_id','parent_id','acquisition_session']:
        if key not in d:continue
        valid=d[d[key].notna() & ~d[key].astype(str).str.lower().isin(['','unknown','missing','nan'])]
        for value,block in valid.groupby(key):
            ids=block.sample_id.tolist()
            for sid in ids[1:]:uf.union(ids[0],sid)
        linkage.append(key)
    if args.reviewed_pairs:
        pairs=pd.read_csv(args.reviewed_pairs,dtype={'sample_id_a':str,'sample_id_b':str}).fillna('')
        status=pairs.verified_related.astype(str).str.strip().str.lower()
        if (~status.isin(['yes','no'])).any():raise ValueError('Every candidate needs explicit yes/no review before grouped reanalysis')
        for _,row in pairs[status=='yes'].iterrows():uf.union(row.sample_id_a,row.sample_id_b)
        linkage.append('verified_duplicate_or_related_pairs')
    d['cluster_id']=[uf.find(s) for s in d.sample_id]
    if not linkage:raise ValueError('No verified grouping information supplied. Do not claim leakage-aware evaluation.')
    if d.cluster_id.nunique()<args.folds:raise ValueError('Too few independent clusters for requested folds')
    outer=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
    rows=[]
    for fold,(rest,test) in enumerate(outer.split(d,d.label,d.cluster_id),1):
        remainder=d.iloc[rest]
        # Fixed first inner split; no choosing a split based on model results.
        inner_k=min(6,remainder.cluster_id.nunique())
        if inner_k<2:raise ValueError('Too few clusters for validation split')
        inner=StratifiedGroupKFold(n_splits=inner_k,shuffle=True,random_state=args.seed+fold)
        train_rel,val_rel=next(inner.split(remainder,remainder.label,remainder.cluster_id))
        train,val=rest[train_rel],rest[val_rel]
        if set(d.iloc[train].label)!=set(d.label):raise ValueError('A grouped training fold lacks a class; adjust protocol explicitly')
        block=d[['sample_id','label','cluster_id']].copy();block['fold']=fold;block['role']=''
        for role,ix in [('Train',train),('Val',val),('Test',test)]:block.loc[ix,'role']=role
        if (block.groupby('cluster_id').role.nunique()>1).any():raise AssertionError('Cluster crosses fold roles')
        rows.append(block)
    out=new_output(args.out);manifest=pd.concat(rows,ignore_index=True)
    manifest.to_csv(out/'grouped_fold_manifest.csv',index=False)
    manifest.groupby(['fold','role','label']).size().rename('n').reset_index().to_csv(out/'grouped_fold_counts.csv',index=False)
    d.to_csv(out/'metadata_with_clusters.csv',index=False)
    write_json(out/'protocol.json',dict(metadata_sha256=sha256(args.metadata),seed=args.seed,folds=args.folds,
        n_images=len(d),n_clusters=d.cluster_id.nunique(),linkage_fields=linkage,
        caveat='NEW folds require new model fitting. Do not evaluate old checkpoints on reassigned test images. pHash absence does not prove patient independence. Validation fraction is approximate; report actual counts.'))


if __name__=='__main__':main()
