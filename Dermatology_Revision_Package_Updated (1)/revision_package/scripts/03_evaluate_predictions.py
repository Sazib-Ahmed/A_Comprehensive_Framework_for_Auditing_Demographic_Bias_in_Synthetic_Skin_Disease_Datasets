"""Evaluate one held-out prediction per image/model; never trains a model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score,
                             log_loss, cohen_kappa_score, matthews_corrcoef)
from audit_common import new_output, write_json, wilson, clopper_pearson, corrected_fold_comparison


def validate_predictions(d, classes):
    required={'sample_id','model','fold','y_true','y_pred'}
    if not required.issubset(d): raise ValueError(f'Prediction columns required: {required}')
    if d[list(required)].isna().any().any(): raise ValueError('Missing required prediction fields')
    if d.duplicated(['model','sample_id']).any():
        raise ValueError('More than one prediction per image/model. Analyze each seed/repeat separately.')
    if len(classes) != len(set(classes)) or len(classes)<2: raise ValueError('Invalid class list')
    if not set(d.y_true).union(d.y_pred).issubset(classes): raise ValueError('Labels absent from classes.json')
    if (d.groupby('sample_id').y_true.nunique()>1).any(): raise ValueError('Model files disagree on truth labels')
    pc=[f'p_{i}' for i in range(len(classes))]
    if any(c in d for c in pc):
        if not all(c in d for c in pc): raise ValueError('Partial probability matrix')
        p=d[pc].to_numpy(float)
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise ValueError('Invalid probabilities')
        if not np.allclose(p.sum(1),1.,atol=1e-5): raise ValueError('Probabilities do not sum to one')
        if not np.array_equal(np.asarray(classes)[p.argmax(1)],d.y_pred.to_numpy()):
            raise ValueError('y_pred inconsistent with probability argmax / class ordering')
    return pc if all(c in d for c in pc) else []


def calibration(d, classes, pc):
    if not pc:return {},[]
    p=d[pc].to_numpy(float); y=np.array([classes.index(s) for s in d.y_true])
    confidence=p.max(1); correct=(p.argmax(1)==y).astype(float)
    bins=np.minimum((confidence*10).astype(int),9); reliability=[]; ece=0.
    for b in range(10):
        ix=bins==b
        n=int(ix.sum())
        if not n:continue
        acc=float(correct[ix].mean()); conf=float(confidence[ix].mean())
        ece += n/len(d)*abs(acc-conf)
        reliability.append(dict(bin=b,lower=b/10,upper=(b+1)/10,n=n,accuracy=acc,mean_confidence=conf))
    onehot=np.eye(len(classes))[y]
    return dict(log_loss=log_loss(y,p,labels=np.arange(len(classes))),
                multiclass_brier_sum=float(((p-onehot)**2).sum(1).mean()),
                top_label_ece_10_equal_width=ece),reliability


def rate_table(d, axis, classes):
    rows=[]
    for g,block in d.groupby(axis,dropna=False):
        for c in classes:
            pos=block.y_true.eq(c); pred=block.y_pred.eq(c)
            tp=int((pos&pred).sum()); fn=int((pos&~pred).sum())
            fp=int((~pos&pred).sum()); tn=int((~pos&~pred).sum())
            plo,phi=wilson(tp,tp+fn); flo,fhi=wilson(fp,fp+tn)
            rows.append(dict(group=str(g),disease=c,n=len(block),tp=tp,fn=fn,fp=fp,tn=tn,
                positive_n=tp+fn,negative_n=fp+tn,tpr=tp/(tp+fn) if tp+fn else np.nan,
                fpr=fp/(fp+tn) if fp+tn else np.nan,tpr95_low=plo,tpr95_high=phi,
                fpr95_low=flo,fpr95_high=fhi))
    return pd.DataFrame(rows)


def gap_bounds(values, bounds):
    if len(values)<2:return (np.nan,np.nan,np.nan)
    value=float(max(values)-min(values))
    low=max(0.,max(b[0] for b in bounds)-min(b[1] for b in bounds))
    high=max(max(0.,bounds[i][1]-bounds[j][0]) for i in range(len(bounds)) for j in range(len(bounds)) if i!=j)
    return value,low,high


def eod_from_rates(rates, classes, minimum_n, minimum_rate_n=1):
    """Explicit OVR EOD = max(TPR range,FPR range), then macro or max over classes.

    A missing denominator is NA, never zero. Each rate range needs >=2 groups.
    Bonferroni-adjusted Clopper-Pearson intervals bound all eligible rate cells.
    These are conservative simultaneous sampling bounds conditional on predictions;
    they exclude training, selection, label-error and clustering uncertainty.
    """
    r=rates[rates.n>=minimum_n]
    n_cells=int((r.positive_n>=minimum_rate_n).sum()+(r.negative_n>=minimum_rate_n).sum())
    alpha=.05/max(n_cells,1); result=[]
    for c in classes:
        z=r[r.disease==c]; metrics={}
        for name,den,num in [('tpr','positive_n','tp'),('fpr','negative_n','fp')]:
            eligible=z[z[den]>=minimum_rate_n]
            values=eligible[name].tolist()
            bounds=[clopper_pearson(int(row[num]),int(row[den]),alpha) for _,row in eligible.iterrows()]
            point,lo,hi=gap_bounds(values,bounds)
            metrics.update({name+'_gap':point,name+'_gap95_low':lo,name+'_gap95_high':hi,
                            name+'_groups':len(eligible)})
        if np.isfinite(metrics['tpr_gap']) and np.isfinite(metrics['fpr_gap']):
            metrics.update(eod=max(metrics['tpr_gap'],metrics['fpr_gap']),
                           eod95_low=max(metrics['tpr_gap95_low'],metrics['fpr_gap95_low']),
                           eod95_high=max(metrics['tpr_gap95_high'],metrics['fpr_gap95_high']))
        else: metrics.update(eod=np.nan,eod95_low=np.nan,eod95_high=np.nan)
        result.append(dict(disease=c,minimum_n=minimum_n,minimum_rate_n=minimum_rate_n,**metrics))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',required=True,nargs='+',help='One or more CSV files with the same class probability order')
    ap.add_argument('--classes',required=True,help='JSON array giving exact p_0,p_1,... class order')
    ap.add_argument('--metadata',required=True,help='02 audit_metadata.csv, one row per sample_id')
    ap.add_argument('--manifest',help='Full sample_id,fold,role for ALL train/validation/test roles')
    ap.add_argument('--compare',help='Two model names separated by comma; both must use same samples/folds')
    ap.add_argument('--minimum-rate-n',type=int,default=1)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.minimum_rate_n<1:raise ValueError('Minimum rate denominator must be >=1')
    d=pd.concat([pd.read_csv(path,dtype={'sample_id':str,'y_true':str,'y_pred':str}) for path in args.predictions],ignore_index=True)
    classes=json.loads(Path(args.classes).read_text());pc=validate_predictions(d,classes)
    md=pd.read_csv(args.metadata,dtype={'sample_id':str})
    if md.sample_id.duplicated().any(): raise ValueError('Metadata sample IDs not unique')
    axes=[c for c in ['age_proxy','gender_proxy','appearance_proxy','composite_proxy','annotation_route','source_status'] if c in md]
    if not axes: raise ValueError('No audit attributes. Run 02_audit_data.py first.')
    if not set(d.sample_id).issubset(set(md.sample_id)): raise ValueError('Predictions missing metadata matches')
    overlap=set(axes).intersection(d.columns)
    if overlap: raise ValueError(f'Ambiguous duplicate attribute columns in predictions: {overlap}')
    d=d.merge(md[['sample_id']+axes],on='sample_id',how='left',validate='many_to_one')
    d[axes]=d[axes].fillna('missing').astype(str)
    manifest=None
    if args.manifest:
        manifest=pd.read_csv(args.manifest,dtype={'sample_id':str})
        if manifest.duplicated(['sample_id','fold']).any():raise ValueError('Duplicate manifest rows')
        checks=d.merge(manifest[['sample_id','fold','role']],on=['sample_id','fold'],how='left',validate='many_to_one')
        if not checks.role.eq('Test').all():raise ValueError('A prediction is not assigned Test in its stated fold')
    out=new_output(args.out)
    overall=[]; diseases=[]; subgroups=[]; rate_rows=[]; eo_rows=[]; sensitivity=[]; reliability=[]
    for model,z in d.groupby('model'):
        cm=confusion_matrix(z.y_true,z.y_pred,labels=classes)
        safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in str(model))
        # Index avoids collisions when sanitized model names coincide.
        mid=list(sorted(d.model.unique())).index(model)
        pd.DataFrame(cm,index=classes,columns=classes).to_csv(out/f'confusion_{mid}_{safe}.csv')
        scores=dict(model=model,n=len(z),accuracy=accuracy_score(z.y_true,z.y_pred),
                    macro_f1=f1_score(z.y_true,z.y_pred,labels=classes,average='macro',zero_division=0),
                    kappa=cohen_kappa_score(z.y_true,z.y_pred,labels=classes),mcc=matthews_corrcoef(z.y_true,z.y_pred))
        scores['accuracy95_low'],scores['accuracy95_high']=wilson(int((z.y_true==z.y_pred).sum()),len(z))
        cal,rel=calibration(z,classes,pc);scores.update(cal)
        reliability.extend([dict(model=model,**r) for r in rel])
        if pc:
            aucs=[]
            for i,c in enumerate(classes):
                target=z.y_true.eq(c).astype(int)
                aucs.append(roc_auc_score(target,z[pc[i]]) if target.nunique()==2 else np.nan)
            scores['auc_classes_estimable']=int(np.isfinite(aucs).sum())
            scores['macro_ovr_auc_all_classes']=float(np.mean(aucs)) if np.isfinite(aucs).all() else np.nan
        overall.append(scores)
        r=rate_table(z.assign(pooled='all'),'pooled',classes)
        for _,row in r.iterrows():
            tp,fp,fn,tn=[int(row[x]) for x in ['tp','fp','fn','tn']]
            precision=tp/(tp+fp) if tp+fp else np.nan
            pl,ph=wilson(tp,tp+fp)
            diseases.append(dict(model=model,disease=row.disease,support=tp+fn,negative_n=fp+tn,
                tp=tp,fn=fn,fp=fp,tn=tn,recall=row.tpr,recall95_low=row.tpr95_low,recall95_high=row.tpr95_high,
                specificity=tn/(tn+fp) if tn+fp else np.nan,
                specificity95_low=1-row.fpr95_high,specificity95_high=1-row.fpr95_low,
                precision=precision,precision95_low=pl,precision95_high=ph,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan))
        scopes=[('all',z)]
        if 'annotation_route' in axes:
            scopes += [('route='+str(route),block) for route,block in z.groupby('annotation_route')]
        for scope,block in scopes:
            for axis in axes:
                groups=[]
                for name,part in block.groupby(axis):
                    n=len(part);k=int((part.y_true==part.y_pred).sum());lo,hi=wilson(k,n)
                    groups.append(dict(group=str(name),n=n,correct=k,accuracy=k/n,wilson95_low=lo,wilson95_high=hi))
                    subgroups.append(dict(model=model,scope=scope,attribute=axis,**groups[-1]))
                rates=rate_table(block,axis,classes)
                rate_rows.extend([dict(model=model,scope=scope,attribute=axis,**row) for row in rates.to_dict('records')])
                for threshold in [1,5,10,15,20,30,50]:
                    eligible=[g for g in groups if g['n']>=threshold]
                    bounds=[clopper_pearson(g['correct'],g['n'],.05/max(1,len(eligible))) for g in eligible]
                    ranks=sorted(eligible,key=lambda g:(g['accuracy'],g['group']))
                    eod=eod_from_rates(rates,classes,threshold,args.minimum_rate_n)
                    eo_rows.extend([dict(model=model,scope=scope,attribute=axis,**e) for e in eod])
                    valid=[e for e in eod if np.isfinite(e['eod'])]
                    sensitivity.append(dict(model=model,scope=scope,attribute=axis,minimum_n=threshold,
                        groups_retained=len(eligible),groups_total=len(groups),
                        image_coverage=sum(g['n'] for g in eligible)/len(block),
                        wga=min((g['accuracy'] for g in eligible),default=np.nan),
                        wga_simultaneous95_low=min((b[0] for b in bounds),default=np.nan),
                        wga_simultaneous95_high=min((b[1] for b in bounds),default=np.nan),
                        eod_classes_estimable=len(valid),
                        macro_eod_all_classes=np.mean([e['eod'] for e in valid]) if len(valid)==len(classes) else np.nan,
                        macro_eod_available=np.mean([e['eod'] for e in valid]) if valid else np.nan,
                        macro_eod_available95_low=np.mean([e['eod95_low'] for e in valid]) if valid else np.nan,
                        macro_eod_available95_high=np.mean([e['eod95_high'] for e in valid]) if valid else np.nan,
                        max_eod_available=max((e['eod'] for e in valid),default=np.nan),
                        max_eod_available95_low=max((e['eod95_low'] for e in valid),default=np.nan),
                        max_eod_available95_high=max((e['eod95_high'] for e in valid),default=np.nan),
                        ranks_low_to_high=json.dumps([g['group'] for g in ranks])))
    for name,rows in [('overall',overall),('disease_metrics',diseases),('subgroup_metrics',subgroups),
                      ('group_class_rates',rate_rows),('eod_per_class',eo_rows),
                      ('support_sensitivity',sensitivity),('reliability_bins',reliability)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    if args.compare:
        a,b=args.compare.split(',')
        left=d[d.model==a];right=d[d.model==b]
        if left.empty or right.empty:raise ValueError('Requested model absent')
        if set(left.sample_id)!=set(right.sample_id):raise ValueError('Paired models have different test sample sets')
        pair=left.merge(right,on='sample_id',validate='one_to_one',suffixes=('_a','_b'))
        if not (pair.fold_a==pair.fold_b).all() or not (pair.y_true_a==pair.y_true_b).all():
            raise ValueError('Paired models disagree on folds/true labels')
        pair['correct_a']=pair.y_true_a==pair.y_pred_a;pair['correct_b']=pair.y_true_b==pair.y_pred_b
        folds=pair.groupby('fold_a')[['correct_a','correct_b']].mean().reset_index().rename(columns={'fold_a':'fold'})
        folds['delta_a_minus_b']=folds.correct_a-folds.correct_b
        folds.to_csv(out/'paired_fold_differences.csv',index=False)
        stats=dict(model_a=a,model_b=b,n=len(pair),mean_fold_delta=float(folds.delta_a_minus_b.mean()),
                   correct_a_wrong_b=int((pair.correct_a&~pair.correct_b).sum()),
                   wrong_a_correct_b=int((~pair.correct_a&pair.correct_b).sum()))
        if manifest is not None:
            ratios=[]
            for f in folds.fold:
                sub=manifest[manifest.fold==f]
                nt=int(sub.role.eq('Train').sum());ne=int(sub.role.eq('Test').sum())
                if nt==0 or ne==0:raise ValueError('Full train/test manifest needed')
                if ne!=int((pair.fold_a==f).sum()):raise ValueError('Prediction coverage differs from manifest test set')
                ratios.append(ne/nt)
            if len(folds)>=3:stats.update(corrected_fold_comparison(folds.delta_a_minus_b,np.mean(ratios)))
        else:stats['inferential_status']='No train/test sizes provided; fold differences only'
        write_json(out/'paired_comparison.json',stats)
    write_json(out/'interpretation.json',dict(
        counts='One held-out prediction per sample/model. Pooling is not averaging subgroup fold accuracies.',
        eod='OVR per disease max(TPR group range,FPR group range). Both ranges need >=2 eligible groups; macro all classes is NA if any class unavailable.',
        missing='Zero positive/negative denominators are NA; no implicit zero rates.',
        uncertainty='Wilson marginal intervals; conservative simultaneous Bonferroni-Clopper-Pearson bounds for WGA/EOD. Conditional on observed predictions and independent image units. No training, tuning, label-error, or patient-cluster uncertainty.',
        exclusions='Threshold curves remove groups from the estimand and must show coverage. Thresholds do not change any predictions.',
        inference='Do not interpret nonsignificance as equivalence. Paired corrected CV tests are approximate and exploratory.'))
    print(f'Wrote evaluation to {out}')


if __name__=='__main__':main()
