"""Cheap, NEW diagnostic classifiers on cached frozen CLS features.

Fits only L2 logistic regression; no backbone or DSAF training. These probes
do not reproduce the original nonlinear fusion heads or training augmentation.
"""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from audit_common import new_output,write_json,sha256


def probe_metadata(d):
    cols=[]
    if 'age_group' in d:
        cols.append(d.age_group.fillna('missing').astype(str))
    elif 'age' in d:
        age=pd.to_numeric(d.age,errors='coerce')
        cols.append(pd.cut(age,[-1,17,29,44,64,200],labels=['child','young_adult','adult','middle_aged','senior']).astype('object').fillna('missing'))
    else:cols.append(pd.Series('missing',index=d.index))
    for primary,fallback in [('gender','dominant_gender'),('skin_tone','dominant_race')]:
        c=d[primary] if primary in d else d[fallback] if fallback in d else pd.Series('missing',index=d.index)
        cols.append(c.fillna('missing').astype(str).str.strip().str.lower())
    return np.stack([np.asarray(c,dtype=str) for c in cols],axis=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',required=True,help='NPZ sample_id and features; independently frozen backbone, no dataset fine-tuning')
    ap.add_argument('--metadata',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--confirm-independent-frozen-features',action='store_true')
    ap.add_argument('--train-control-repeats',type=int,default=0)
    ap.add_argument('--seed',type=int,default=20261004);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if not args.confirm_independent_frozen_features:raise ValueError('Confirm feature extractor was not trained/fine-tuned on these disease labels or held-out images.')
    if args.train_control_repeats<0:raise ValueError('Negative repeat count')
    cache=np.load(args.features,allow_pickle=False)
    ids=cache['sample_id'].astype(str);features=cache['features']
    if len(set(ids))!=len(ids) or features.ndim!=2 or len(features)!=len(ids) or not np.isfinite(features).all():raise ValueError('Invalid feature cache')
    md=pd.read_csv(args.metadata,dtype={'sample_id':str})
    if md.sample_id.duplicated().any() or not set(ids).issubset(set(md.sample_id)):raise ValueError('Metadata ID mismatch')
    md=md.set_index('sample_id').loc[ids].reset_index()
    if md.label.isna().any():raise ValueError('Missing disease labels')
    manifest=pd.read_csv(args.manifest,dtype={'sample_id':str})
    if manifest.duplicated(['sample_id','fold']).any():raise ValueError('Duplicate manifest rows')
    if set(manifest.sample_id)!=set(ids):raise ValueError('Manifest and feature cache have different sample sets')
    if not set(manifest.role).issubset({'Train','Val','Test'}):raise ValueError('Invalid roles')
    if manifest[manifest.role=='Test'].sample_id.value_counts().reindex(ids,fill_value=0).ne(1).any():
        raise ValueError('Every image must be held out exactly once')
    classes=sorted(md.label.astype(str).unique());y=np.array([classes.index(str(v)) for v in md.label])
    metadata=probe_metadata(md);rng=np.random.default_rng(args.seed)
    out=new_output(args.out);predictions=[];logs=[]
    for fold,roles in manifest.groupby('fold'):
        roles=roles.set_index('sample_id').reindex(ids)
        if roles.role.isna().any():raise ValueError('Each fold needs all samples and their roles')
        tr=np.flatnonzero(roles.role.eq('Train'));te=np.flatnonzero(roles.role.eq('Test'))
        if set(y[tr])!=set(range(len(classes))):raise ValueError('Training fold lacks a disease class')
        scaler=StandardScaler().fit(features[tr]);xtr=scaler.transform(features[tr]);xte=scaler.transform(features[te])
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(metadata[tr])
        mtr=encoder.transform(metadata[tr]);mte=encoder.transform(metadata[te])
        inputs=[('probe_image_only',xtr,xte),('probe_metadata_only',mtr,mte),
                ('probe_concat',np.c_[xtr,mtr],np.c_[xte,mte])]
        for repeat in range(args.train_control_repeats):
            for mode in ['joint_tuple','independent_columns']:
                permuted=metadata[tr].copy()
                if mode=='joint_tuple':permuted=permuted[rng.permutation(len(tr))]
                else:
                    for col in range(permuted.shape[1]):permuted[:,col]=permuted[rng.permutation(len(tr)),col]
                # Test metadata stays authentic. This asks what is learned when train metadata is uninformative.
                inputs.append((f'probe_{mode}_train_randomized_r{repeat}',np.c_[xtr,encoder.transform(permuted)],np.c_[xte,mte]))
        for name,a,b in inputs:
            model=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,random_state=args.seed)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(a,y[tr])
            p=model.predict_proba(b)
            if not np.array_equal(model.classes_,np.arange(len(classes))):raise ValueError('Unexpected probability class order')
            pred=p.argmax(1)
            for j,i in enumerate(te):
                predictions.append(dict(sample_id=ids[i],model=name,fold=fold,y_true=classes[y[i]],y_pred=classes[pred[j]],
                                   **{f'p_{c}':float(p[j,c]) for c in range(len(classes))}))
            logs.append(dict(model=name,fold=fold,n_train=len(tr),n_test=len(te),n_features=a.shape[1],
                             accuracy=float((pred==y[te]).mean()),iterations=int(model.n_iter_.max())))
    pd.DataFrame(predictions).to_csv(out/'probe_oof_predictions.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'probe_fold_metrics.csv',index=False)
    write_json(out/'classes.json',classes)
    write_json(out/'protocol.json',dict(feature_file_sha256=sha256(args.features),manifest_sha256=sha256(args.manifest),
        metadata_sha256=sha256(args.metadata),seed=args.seed,sklearn_version=sklearn.__version__,C=1.,
        models='L2 multiclass logistic probes; fixed hyperparameters, no test-driven selection; train-only image scaling and metadata vocabulary',
        feature_requirement='Backbone independent of this dataset fine-tuning. Old held-out feature caches invalid for new folds if backbone was fine-tuned.',
        exclusions='Validation rows are not used for fitting these fixed-configuration probes.',
        caveat='NEW diagnostic experiment; not matched-architecture ablations of original DSAF/concat; not evidence that source demographics cause disease. No fresh DINO training.'))
    print(f'Fitted {len(logs)} small classifiers; wrote {out}')


if __name__=='__main__':main()
