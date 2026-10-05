"""Recover verified aggregate results without loading images or training models."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from audit_common import (new_output, write_json, sha256, parse_saved_counts, wilson,
                          clopper_pearson, corrected_fold_comparison)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve(); out = new_output(args.out)
    try: commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = 'unavailable'
    provenance = {'commit':commit, 'files':{}, 'scope':'Released summaries only; no identity-level pairing verified'}
    summary, fold_rows, group_rows, sensitivity, historical = [], [], [], [], []
    models = {}
    for p in sorted(repo.glob('*/kfold_performance.csv')):
        model = p.parent.name
        provenance['files'][str(p.relative_to(repo))] = sha256(p)
        d = pd.read_csv(p)
        if d.fold.duplicated().any() or set(d.fold) != set(range(1,8)):
            raise ValueError(f'Expected seven unique folds: {p}')
        d = d.sort_values('fold'); models[model] = d
        for metric in ['accuracy','roc_auc_macro','kappa','mcc','wga_all','wga_filtered']:
            v = d[metric].to_numpy(float)
            summary.append(dict(model=model, metric=metric, mean=v.mean(), sample_sd=v.std(ddof=1),
                                population_sd=v.std(ddof=0), n_folds=len(v)))
        pooled = defaultdict(lambda: defaultdict(lambda:[0,0]))
        all_attrs = None
        for _, row in d.iterrows():
            counts = parse_saved_counts(row.pooled_groups)
            totals = {a:sum(n for k,n in g.values()) for a,g in counts.items()}
            correct = {a:sum(k for k,n in g.values()) for a,g in counts.items()}
            if len(set(totals.values())) != 1 or len(set(correct.values())) != 1:
                raise ValueError(f'Attribute accounting mismatch {model} fold {row.fold}')
            n = next(iter(totals.values())); k = next(iter(correct.values()))
            if not np.isclose(k/n, row.accuracy, atol=1e-12): raise ValueError('Accuracy/count mismatch')
            fold_rows.append(dict(model=model, fold=int(row.fold), n_test=n, correct=k,
                                  accuracy=row.accuracy, roc_auc_macro=row.roc_auc_macro))
            for attr, groups in counts.items():
                for name,(k,n) in groups.items():
                    pooled[attr][name][0] += k; pooled[attr][name][1] += n
                add_sensitivity(sensitivity, model, f'fold_{int(row.fold)}', attr, groups)
            all_attrs = counts.keys()
        for attr in all_attrs:
            for name,(k,n) in pooled[attr].items():
                lo,hi = wilson(k,n)
                group_rows.append(dict(model=model, attribute=attr, raw_saved_group=name,
                                  correct=k, count=n, accuracy=k/n, wilson95_low=lo, wilson95_high=hi,
                                  label_warning='Legacy age display: 18-59 actually combines bins through 64; 60+ uses senior >64. Verify raw CSV.'))
            add_sensitivity(sensitivity, model, 'pooled', attr, pooled[attr])
    pd.DataFrame(summary).to_csv(out/'model_summary_recomputed.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_metrics_verified_counts.csv', index=False)
    pd.DataFrame(group_rows).to_csv(out/'subgroup_counts_and_wilson.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out/'support_sensitivity.csv', index=False)
    # Recorded notebook outputs are historical evidence, not freshly reproduced model results.
    for p in sorted(repo.rglob('*.ipynb')):
        if any(s.startswith('.') for s in p.relative_to(repo).parts): continue
        provenance['files'][str(p.relative_to(repo))] = sha256(p)
        nb = json.loads(p.read_text(encoding='utf-8'))
        for ci,c in enumerate(nb['cells']):
            texts = []
            for ob in c.get('outputs', []):
                text = ob.get('text', ob.get('data',{}).get('text/plain',[]))
                texts.append(''.join(text) if isinstance(text,list) else text)
            txt = '\n'.join(texts)
            for m in re.finditer(r'--- Overall Scalar Metrics ---\s*Accuracy\s*:\s*([\d.]+)\s*Macro ROC AUC\s*:\s*([\d.]+)\s*Cohen\'s Kappa\s*:\s*([\d.]+)\s*Matthews CorrCoef\s*:\s*([\d.]+)',txt):
                historical.append(dict(notebook=str(p.relative_to(repo)), cell=ci,
                                  accuracy=float(m[1]), auc=float(m[2]), kappa=float(m[3]),mcc=float(m[4])))
            if 'full_Dataset' in str(p) and 'unfreeze' not in str(p):
                reports = re.findall(r'precision\s+recall\s+f1-score\s+support[\s\S]*?weighted avg[^\n]*',txt)
                for ri,report in enumerate(reports):
                    (out/f'{p.stem}_cell{ci}_report{ri}.txt').write_text(report+'\n', encoding='utf-8')
    pd.DataFrame(historical).to_csv(out/'historical_notebook_metrics.csv', index=False)
    a = models.get('metadata_Multimodal_DINOv3_Concat_KFold')
    b = models.get('metadata_Multimodal_DINOv3_DSAF_KFold')
    if a is not None and b is not None:
        pair = a[['fold','accuracy']].merge(b[['fold','accuracy']], on='fold', validate='one_to_one',suffixes=('_concat','_dsaf'))
        pair['delta_concat_minus_dsaf'] = pair.accuracy_concat-pair.accuracy_dsaf
        pair['pairing_status'] = 'FOLD NUMBER ONLY; SAMPLE IDs UNAVAILABLE'
        pair.to_csv(out/'provisional_fold_comparison.csv', index=False)
        dif = pair.delta_concat_minus_dsaf.to_numpy()
        # Exact counts from N=565, seven-fold splitter and 17.5% remainder validation.
        ratio = float(np.mean([81/399]*5+[80/400]*2))
        stats = corrected_fold_comparison(dif, ratio)
        stats['sign_flip_two_sided_p_exploratory'] = float(np.mean([
            abs(np.mean(dif*np.asarray(signs))) >= abs(dif.mean())-1e-12
            for signs in itertools.product([-1,1], repeat=len(dif))]))
        stats['publication_status'] = 'PROVISIONAL: validate identical sample IDs/labels per fold before reporting as paired inference'
        stats['limitations'] = 'Overlapping CV training sets; one CV run; configuration-selection history unresolved; no equivalence claim. Sign-flip independence is not guaranteed.'
        write_json(out/'provisional_paired_statistics.json', stats)
    write_json(out/'provenance.json', provenance)
    print(f'Wrote aggregate audit to {out}; no training or inference performed.')


def add_sensitivity(rows, model, scope, attr, groups):
    total = sum(n for k,n in groups.values())
    for threshold in [1,5,10,15,20,30,50]:
        selected = {g:(k,n) for g,(k,n) in groups.items() if n >= threshold}
        rank = sorted(selected, key=lambda g:(selected[g][0]/selected[g][1],g))
        bounds = [clopper_pearson(k,n,.05/max(1,len(selected))) for k,n in selected.values()]
        rows.append(dict(model=model, scope=scope, attribute=attr, minimum_n=threshold,
                         groups_retained=len(selected), groups_excluded=len(groups)-len(selected),
                         images_retained=sum(n for k,n in selected.values()), images_total=total,
                         coverage=sum(n for k,n in selected.values())/total,
                         wga=min((k/n for k,n in selected.values()), default=np.nan),
                         wga_simultaneous95_low=min((v[0] for v in bounds),default=np.nan),
                         wga_simultaneous95_high=min((v[1] for v in bounds),default=np.nan),
                         ranks_low_to_high=json.dumps(rank),
                         caution='Conditional binomial bounds; no patient clusters or training variability available'))


if __name__ == '__main__': main()
