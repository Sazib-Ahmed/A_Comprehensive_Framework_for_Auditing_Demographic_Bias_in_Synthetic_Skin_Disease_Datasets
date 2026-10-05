"""Audit actual image files, metadata accounting, route composition and duplicates."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from audit_common import (new_output, write_json, sha256, audit_proxy_columns,
                          resolve_image, build_flat_image_index)

EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}


def perceptual_hash(im):
    a = np.asarray(im.convert('L').resize((32,32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(a, type=2, norm='ortho')[:8,:8].ravel()
    bits = low > np.median(low[1:])
    return sum(int(v) << i for i,v in enumerate(bits))


def duplicate_pairs(images, max_distance):
    rows=[]; good=images[images.decode_ok]
    records=good.to_dict('records')
    for i,a in enumerate(records):
        for b in records[i+1:]:
            exact_bytes=a['sha256']==b['sha256']
            exact_pixels=a['pixel_sha256']==b['pixel_sha256']
            distance=(int(a['phash'],16)^int(b['phash'],16)).bit_count()
            if exact_bytes or exact_pixels or distance <= max_distance:
                rows.append(dict(sample_id_a=a['sample_id'],sample_id_b=b['sample_id'],
                           byte_identical=exact_bytes,pixel_identical=exact_pixels,
                           phash_distance=distance,
                           archive_split_a=a['archive_split'],archive_split_b=b['archive_split'],
                           label_a=a['archive_label'],label_b=b['archive_label'],
                           verified_related='yes' if exact_pixels else '',
                           review_reason='exact decoded pixels' if exact_pixels else 'CANDIDATE ONLY; visual/source review required'))
    return pd.DataFrame(rows, columns=['sample_id_a','sample_id_b','byte_identical','pixel_identical',
        'phash_distance','archive_split_a','archive_split_b','label_a','label_b','verified_related','review_reason'])


def cross_split_pairs(pairs, manifest):
    required={'sample_id','fold','role'}
    if not required.issubset(manifest): raise ValueError(f'Fold manifest needs {required}')
    if manifest.duplicated(['sample_id','fold']).any(): raise ValueError('Duplicate sample/fold manifest rows')
    if not set(manifest.role).issubset({'Train','Val','Test'}): raise ValueError('Roles must be Train/Val/Test')
    a=manifest.rename(columns={'sample_id':'sample_id_a','role':'role_a'})
    b=manifest.rename(columns={'sample_id':'sample_id_b','role':'role_b'})
    j=pairs.merge(a,on='sample_id_a').merge(b,on=['sample_id_b','fold'])
    return j[j.role_a != j.role_b]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--metadata',required=True)
    ap.add_argument('--strip-prefix',default='')
    ap.add_argument('--flat-images',action='store_true',
                    help='Images are stored in one directory; match metadata by unique basename after an ambiguity check')
    ap.add_argument('--route-manifest',help='Optional sample_id,annotation_route with documented provenance')
    ap.add_argument('--fold-manifest',help='Actual sample_id,fold,role assignments; optional')
    ap.add_argument('--reconstruct-legacy-splits',action='store_true',
                    help='Create the released seed-42 image-level folds after confirming original row order')
    ap.add_argument('--confirm-original-row-order',action='store_true')
    ap.add_argument('--phash-distance',type=int,default=6)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if not 0 <= args.phash_distance <= 64: raise ValueError('pHash distance must be 0..64')
    if args.fold_manifest and args.reconstruct_legacy_splits:
        raise ValueError('Use either --fold-manifest or --reconstruct-legacy-splits, not both')
    if args.reconstruct_legacy_splits and not args.confirm_original_row_order:
        raise ValueError('Legacy reconstruction requires --confirm-original-row-order')
    root=Path(args.data_root).resolve(); out=new_output(args.out)
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXTS: continue
        rel=p.relative_to(root)
        record=dict(sample_id=rel.as_posix(),filepath=rel.as_posix(),bytes=p.stat().st_size,
                    sha256=sha256(p),archive_split=rel.parts[0] if len(rel.parts)>=3 else 'unknown',
                    archive_label=rel.parts[-2] if len(rel.parts)>=2 else 'unknown',decode_ok=False)
        try:
            with Image.open(p) as source:
                im=source.convert('RGB'); im.load()
                record.update(decode_ok=True,width=im.width,height=im.height,
                    pixel_sha256=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(),
                    phash=f'{perceptual_hash(im):016x}',
                    grayscale_mean=float(np.asarray(im.convert('L'),dtype=float).mean()))
        except (OSError,ValueError,Image.DecompressionBombError) as e:
            record['decode_error']=type(e).__name__+': '+str(e)
        rows.append(record)
    if not rows: raise ValueError('No images found at supplied data root')
    images=pd.DataFrame(rows)
    images.to_csv(out/'image_inventory.csv',index=False)
    pairs=duplicate_pairs(images,args.phash_distance)
    # No source/parent/patient relationship is inferred from perceptual similarity.
    raw=pd.read_csv(args.metadata)
    flat_index=build_flat_image_index(root) if args.flat_images else None
    resolved=[resolve_image(row,root,args.strip_prefix,flat_index=flat_index) for _,row in raw.iterrows()]
    if 'original_filepath' not in raw:
        raw.insert(0,'original_filepath',raw.get('filepath',pd.Series('',index=raw.index)).astype(str))
    raw['sample_id']=[sid for _,sid in resolved]
    raw['filepath']=[sid for _,sid in resolved]
    collisions=raw[raw.sample_id.duplicated(keep=False)]
    collisions.to_csv(out/'repeated_metadata_rows.csv',index=False)
    if len(collisions):
        raise ValueError('Repeated image IDs in metadata. Resolve explicitly; no automatic deduplication performed.')
    lookup=raw.set_index('sample_id')
    for side in ['a','b']:
        ids=pairs[f'sample_id_{side}']
        pairs[f'metadata_split_{side}']=ids.map(lookup['split']) if 'split' in lookup else 'unknown'
        pairs[f'metadata_label_{side}']=ids.map(lookup['label']) if 'label' in lookup else 'unknown'
    pairs.to_csv(out/'duplicate_candidates_for_review.csv',index=False)
    if args.route_manifest:
        routes=pd.read_csv(args.route_manifest, dtype={'sample_id':str})
        if routes.sample_id.duplicated().any(): raise ValueError('Duplicate route IDs')
        if 'annotation_route' in raw: raise ValueError('Route column already present; reconcile sources explicitly')
        if not set(routes.sample_id).issubset(set(images.sample_id)):
            raise ValueError('Route manifest contains IDs outside image inventory')
        raw=raw.merge(routes,on='sample_id',how='left',validate='one_to_one')
    d=audit_proxy_columns(raw)
    original_order=d.sample_id.astype(str).tolist()
    d=d.merge(images.drop(columns=['filepath']),on='sample_id',how='left',sort=False,validate='one_to_one')
    if d.sample_id.astype(str).tolist()!=original_order:
        raise AssertionError('Metadata row order changed during image-inventory join')
    if d.decode_ok.isna().any() or not d.decode_ok.all(): raise ValueError('Metadata includes unscanned or unreadable images')
    if 'label' not in d or d.label.isna().any(): raise ValueError('Metadata must include verified disease label')
    d.to_csv(out/'audit_metadata.csv',index=False)
    for left,right in [('label','annotation_route'),('label','appearance_proxy'),('label','gender_proxy'),
                       ('label','age_proxy'),('annotation_route','appearance_proxy'),
                       ('annotation_route','gender_proxy'),('annotation_route','age_proxy'),
                       ('source_status','appearance_proxy'),('source_status','gender_proxy'),
                       ('source_status','age_proxy'),('annotation_route','source_status'),
                       ('annotation_route','body_site'),('body_site','appearance_proxy')]:
        pd.crosstab(d[left],d[right],dropna=False).to_csv(out/f'counts_{left}_by_{right}.csv')
    d.groupby('annotation_route')[['width','height','bytes','grayscale_mean']].agg(
        ['count','median','min','max']).to_csv(out/'image_characteristics_by_route.csv')
    mappings=[]
    for source_col,final_col in [('appearance_raw_normalized','appearance_proxy'),('gender_raw_normalized','gender_proxy')]:
        for (raw_label,final_label),n in d.groupby([source_col,final_col],dropna=False).size().items():
            mappings.append(dict(attribute=final_col,raw_label=raw_label,final_label=final_label,n=n))
    pd.DataFrame(mappings).to_csv(out/'observed_mapping.csv',index=False)
    # CSV counts should be reconstructed from original files, not manually forced to 565.
    accounting=[dict(stage='original image files',n=len(images)),
                dict(stage='decode success',n=int(images.decode_ok.sum())),
                dict(stage='decode failures (not automatically excluded)',n=int((~images.decode_ok).sum())),
                dict(stage='metadata rows',n=len(d)),dict(stage='unique proxy subset images',n=d.sample_id.nunique())]
    for c in ['age_proxy','gender_proxy','appearance_proxy','annotation_route','source_status','body_site']:
        for value,n in d[c].value_counts(dropna=False).items(): accounting.append(dict(stage=c+': '+str(value),n=int(n)))
    pd.DataFrame(accounting).to_csv(out/'dataset_accounting.csv',index=False)
    manifest=None
    if args.fold_manifest:
        manifest=pd.read_csv(args.fold_manifest, dtype={'sample_id':str})
    elif args.reconstruct_legacy_splits:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
        y=d.label.astype(str).to_numpy();outer=StratifiedKFold(n_splits=7,shuffle=True,random_state=42)
        blocks=[]
        for fold,(rest,test) in enumerate(outer.split(np.arange(len(d)),y),1):
            inner=StratifiedShuffleSplit(n_splits=1,test_size=.175,random_state=42+fold)
            train_rel,val_rel=next(inner.split(rest,y[rest]))
            block=d[['sample_id','label']].copy();block['fold']=fold;block['role']=''
            block.loc[rest[train_rel],'role']='Train';block.loc[rest[val_rel],'role']='Val';block.loc[test,'role']='Test'
            blocks.append(block)
        manifest=pd.concat(blocks,ignore_index=True)
        manifest.to_csv(out/'reconstructed_fold_manifest.csv',index=False)
        manifest.groupby(['fold','role','label']).size().rename('n').reset_index().to_csv(out/'fold_counts.csv',index=False)
        write_json(out/'split_provenance.json',dict(metadata_sha256=sha256(args.metadata),split_seed=42,
            validation_seed='42+fold',validation_fraction_of_remainder=.175,
            limitation='Reconstruction, not historically saved identity evidence. Original CSV row order was explicitly confirmed. No patient/duplicate grouping.'))
    if manifest is not None:
        if not set(manifest.sample_id).issubset(set(images.sample_id)): raise ValueError('Fold IDs not in image inventory')
        cross_split_pairs(pairs,manifest).to_csv(out/'duplicate_candidates_crossing_roles.csv',index=False)
        relations=[]
        for key in ['patient_id','case_id','parent_id','acquisition_session']:
            if key not in d: continue
            known=d[d[key].notna() & ~d[key].astype(str).str.lower().isin(['','unknown','missing','nan'])]
            z=manifest.merge(known[['sample_id',key]],on='sample_id',validate='many_to_one')
            for (fold,entity),block in z.groupby(['fold',key]):
                if block.role.nunique()>1:
                    relations.append(dict(identifier=key,value=entity,fold=fold,roles='|'.join(sorted(block.role.unique())),n=len(block)))
        pd.DataFrame(relations,columns=['identifier','value','fold','roles','n']).to_csv(out/'known_related_cases_crossing_roles.csv',index=False)
    write_json(out/'audit_notes.json',dict(metadata_sha256=sha256(args.metadata),
        raw_files=len(images),proxy_rows=len(d),phash_distance=args.phash_distance,
        image_path_mode='unique basename lookup in explicitly flat image directory' if args.flat_images else 'metadata path relative to data root',
        duplicate_interpretation='pHash candidates require review; no duplicates removed; absence of matches does not exclude common source cases',
        source_interpretation='Unknown stays unknown. Real/synthetic and route are not inferred from image appearance or filenames.',
        age_interpretation='Audit bins 0-17,18-59,60+ derive from numeric age. Do not replace model input bins without retraining.',
        missing_id_fields=[c for c in ['patient_id','case_id','parent_id','acquisition_session'] if c not in d]))
    print(f'Wrote {len(images)} image records and {len(d)} metadata records to {out}')


if __name__=='__main__': main()
