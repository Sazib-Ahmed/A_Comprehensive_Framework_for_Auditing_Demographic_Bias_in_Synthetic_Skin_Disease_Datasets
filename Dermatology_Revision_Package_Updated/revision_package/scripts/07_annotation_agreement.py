"""Summarize two genuinely independent reannotations; cannot recover old agreement."""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from audit_common import new_output,write_json,wilson


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rater-a',required=True);ap.add_argument('--rater-b',required=True)
    ap.add_argument('--confirm-independent',action='store_true')
    ap.add_argument('--columns',default='appearance_proxy,gender_proxy,age_proxy')
    ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--seed',type=int,default=20261004)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    if not args.confirm_independent:raise ValueError('Existing consensus labels are not independent raters.')
    if args.bootstrap<1:raise ValueError('Positive bootstrap count required')
    a=pd.read_csv(args.rater_a,dtype=str);b=pd.read_csv(args.rater_b,dtype=str)
    if a.sample_id.duplicated().any() or b.sample_id.duplicated().any():raise ValueError('Duplicate image IDs')
    if set(a.sample_id)!=set(b.sample_id):raise ValueError('Raters must label same preselected subset; do not drop unmatched rows silently')
    d=a.merge(b,on='sample_id',suffixes=('_a','_b'),validate='one_to_one');out=new_output(args.out)
    rng=np.random.default_rng(args.seed);summary=[]
    for col in args.columns.split(','):
        x=d[col+'_a'];y=d[col+'_b'];complete=x.notna()&y.notna()&x.str.strip().ne('')&y.str.strip().ne('')
        xx=x[complete].str.strip().to_numpy();yy=y[complete].str.strip().to_numpy();n=len(xx)
        pd.crosstab(x.fillna('MISSING'),y.fillna('MISSING'),dropna=False).to_csv(out/(col+'_disagreement_table.csv'))
        if n==0:
            summary.append(dict(attribute=col,n=0,missing_pairs=len(d)));continue
        # 'unclear' is an observed category, never silently deleted. Agreement != label validity.
        categories=np.unique(np.r_[xx,yy]);kappa=cohen_kappa_score(xx,yy,labels=categories) if len(categories)>1 else np.nan
        boot=[]
        for _ in range(args.bootstrap):
            ix=rng.integers(0,n,n)
            if len(np.unique(np.r_[xx[ix],yy[ix]]))>1:
                val=cohen_kappa_score(xx[ix],yy[ix],labels=categories)
                if np.isfinite(val):boot.append(val)
        lo,hi=wilson(int((xx==yy).sum()),n)
        # Avoid a misleading CI when most bootstrap samples are degenerate.
        klo,khi=np.quantile(boot,[.025,.975]) if len(boot)>=.9*args.bootstrap else (np.nan,np.nan)
        summary.append(dict(attribute=col,n=n,missing_pairs=len(d)-n,agreement=float((xx==yy).mean()),
            agreement95_low=lo,agreement95_high=hi,kappa=kappa,kappa_boot95_low=klo,kappa_boot95_high=khi,
            bootstrap_valid=len(boot),bootstrap_requested=args.bootstrap))
    pd.DataFrame(summary).to_csv(out/'agreement_summary.csv',index=False)
    write_json(out/'interpretation.json',dict(scope='New independent subset audit; not retrospective reliability of jointly constructed labels',
        limitation='Agreement does not establish demographic/clinical validity. Bootstrap assumes independent sampled images. Stratified subset oversampling changes prevalence-dependent kappa; report sampling design and strata.',
        blindness='Document independence and blinding to model predictions/previous labels. Do not infer identity or phototype from agreement.'))


if __name__=='__main__':main()
