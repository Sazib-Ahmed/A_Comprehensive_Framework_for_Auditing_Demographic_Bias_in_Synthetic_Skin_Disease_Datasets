"""Recover predictions from the RELEASED DINOv3 fold checkpoints. No training.

Requires the author's original ordered metadata CSV, full saved state_dicts,
and the exact local Hugging Face backbone config. Uses first notebook cell
definitions only; does not execute a notebook or any training cells.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from audit_common import (new_output, write_json, sha256, parse_saved_counts,
                          resolve_image, build_flat_image_index)


def load_definitions(repo, kind, torch, nn, AutoModel, backbone_config, gate_scale, meta_dropout):
    dirname='metadata_Multimodal_DINOv3_'+('Concat' if kind=='concat' else 'DSAF')+'_KFold'
    notebook=repo/dirname/(dirname+'.ipynb')
    nb=json.loads(notebook.read_text(encoding='utf-8'))
    source=''.join(next(c for c in nb['cells'] if c['cell_type']=='code')['source'])
    names={'standardize_metadata_columns','MetaVocab','MetadataEncoder','build_kfold_splits'}
    names |= {'DINOv3MultimodalConcat'} if kind=='concat' else {'DSAFHeadMultimodal','DINOv3Multimodal'}
    tree=ast.parse(source)
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != names: raise ValueError('Unexpected repository definitions/version')
    class LocalModel:
        @staticmethod
        def from_pretrained(unused_name):
            # Full original checkpoint supplies ALL weights. No model download.
            return AutoModel.from_config(backbone_config)
    env={'np':np,'pd':pd,'os':os,'torch':torch,'nn':nn,'AutoModel':LocalModel,
         'HAVE_TRANSFORMERS':True, 'StratifiedKFold':StratifiedKFold, 'StratifiedShuffleSplit':StratifiedShuffleSplit,
         'config':SimpleNamespace(PARTIALLY_UNFREEZE=False,GATE_SCALE=gate_scale,META_DROPOUT=meta_dropout)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(notebook),'exec'),env)
    return env,notebook,repo/dirname/'kfold_performance.csv'


def verify_count_fingerprint(df, reference):
    for f,part in df.groupby('fold'):
        observed=part[part.role=='Test']
        ref=parse_saved_counts(reference.loc[reference.fold==f,'pooled_groups'].iloc[0])
        age=observed.age_group.map({'child':'0-17','young_adult':'18-59','adult':'18-59','middle_aged':'18-59','senior':'60+'}).fillna('unknown')
        values={'gender':observed.gender.astype(str),'age':age,
                'skin_tone':observed.skin_tone.astype(str),
                'composite':age+'|'+observed.gender.astype(str)+'|'+observed.skin_tone.astype(str)}
        for attr,v in values.items():
            if v.value_counts().to_dict() != {g:n for g,(k,n) in ref[attr].items()}:
                raise ValueError(f'Fold {f} {attr} counts differ from released CSV. Wrong metadata/order/config; stop.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True);ap.add_argument('--metadata',required=True)
    ap.add_argument('--data-root',required=True);ap.add_argument('--strip-prefix',default='')
    ap.add_argument('--flat-images',action='store_true',
                    help='Match metadata paths to a flat image directory by unique basename')
    ap.add_argument('--kind',choices=['concat','dsaf'],required=True)
    ap.add_argument('--checkpoint-dir',required=True)
    ap.add_argument('--backbone-config',required=True,help='Exact LOCAL HF model config directory/file used for training')
    ap.add_argument('--confirm-original-row-order',action='store_true')
    ap.add_argument('--gate-scale',type=float,default=.5,help='Historical DSAF value. Not present in state_dict.')
    ap.add_argument('--meta-dropout',type=float,default=.3)
    ap.add_argument('--permutations',type=int,default=20)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--device',default='cuda');ap.add_argument('--seed',type=int,default=20261004)
    ap.add_argument('--save-features',action='store_true',help='Save CLS features for cheap diagnostic probes; verifies identical frozen backbone weights across folds')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if not args.confirm_original_row_order:
        raise ValueError('Recovery requires the exact original CSV row order. See README, then pass --confirm-original-row-order.')
    if args.permutations<0 or args.batch_size<1:raise ValueError('Invalid permutations/batch size')
    import torch
    from torch import nn
    from torchvision import transforms
    from PIL import Image
    from transformers import AutoModel,AutoConfig
    from sklearn.metrics import log_loss
    cfg=AutoConfig.from_pretrained(args.backbone_config,local_files_only=True)
    if int(cfg.hidden_size)!=768: raise ValueError('Released backbone has hidden size 768')
    env,notebook,reference_path=load_definitions(Path(args.repo),args.kind,torch,nn,AutoModel,cfg,args.gate_scale,args.meta_dropout)
    raw=pd.read_csv(args.metadata)
    flat_index=build_flat_image_index(args.data_root) if args.flat_images else None
    resolved=[resolve_image(row,args.data_root,args.strip_prefix,flat_index=flat_index) for _,row in raw.iterrows()]
    ids=[v[1] for v in resolved]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate image IDs in metadata; do not silently drop rows')
    df=env['standardize_metadata_columns'](raw)
    df['sample_id']=ids;df['resolved_path']=[str(v[0]) for v in resolved]
    classes=sorted(df.label.astype(str).unique());label2id={c:i for i,c in enumerate(classes)}
    if len(classes)!=5:raise ValueError('Expected five classes')
    folds=env['build_kfold_splits'](df,n_splits=7,seed=42,val_within_remainder=.175)
    manifest=pd.concat([f.assign(fold=i+1,role=f.split) for i,f in enumerate(folds)],ignore_index=True)
    reference=pd.read_csv(reference_path)
    verify_count_fingerprint(manifest,reference)
    out=new_output(args.out)
    # This is RECONSTRUCTED, not a historically saved split file. Provenance states this.
    manifest[['sample_id','fold','role','label']].to_csv(out/'reconstructed_fold_manifest.csv',index=False)
    write_json(out/'classes.json',classes)
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                           transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    rng=np.random.default_rng(args.seed);predictions=[];perturbations=[];runlog=[]
    feature_parts=[];feature_ids=[];backbone_hash=None
    for fold_no,fold in enumerate(folds,1):
        train=fold[fold.split=='Train'];test=fold[fold.split=='Test'].reset_index(drop=True)
        vocab=env['MetaVocab'](train)
        if args.kind=='concat':
            model=env['DINOv3MultimodalConcat']('local',5,vocab,128,256,args.meta_dropout)
            fname=f'best_model_fold_{fold_no}.pth'
        else:
            model=env['DINOv3Multimodal']('local',5,vocab)
            fname=f'best_multimodal_dsaf_fold_{fold_no}.pth'
        checkpoint=Path(args.checkpoint_dir)/fname
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not isinstance(state,dict):raise ValueError('Expected a full state_dict, not a serialized model')
        # THOP may leave counters in a checkpoint; no trained keys are discarded.
        counters=[k for k in state if k.endswith('.total_ops') or k.endswith('.total_params') or k in ['total_ops','total_params']]
        for k in counters:del state[k]
        model.load_state_dict(state,strict=True);del state
        backbone=model.backbone if args.kind=='concat' else model.dinov3
        if args.save_features:
            h=hashlib.sha256()
            for name,tensor in sorted(backbone.state_dict().items()):
                h.update(name.encode());h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            digest=h.hexdigest()
            if backbone_hash is not None and backbone_hash!=digest:
                raise ValueError('Backbone weights differ across folds; a global fixed-feature probe cache would be invalid')
            backbone_hash=digest
        model.to(args.device).eval();features=[]
        with torch.inference_mode():
            for start in range(0,len(test),args.batch_size):
                batch=[]
                for p in test.resolved_path.iloc[start:start+args.batch_size]:
                    with Image.open(p) as im:batch.append(tfm(im.convert('RGB')))
                x=torch.stack(batch).to(args.device)
                features.append(backbone(pixel_values=x).last_hidden_state.cpu())
        hidden=torch.cat(features);del features
        metadata={c:np.array([vocab.encode_row(row)[c] for _,row in test.iterrows()]) for c in vocab.cols}
        def predict(meta):
            probs=[]
            with torch.inference_mode():
                for start in range(0,len(test),args.batch_size):
                    token=hidden[start:start+args.batch_size].to(args.device)
                    ix={c:torch.as_tensor(v[start:start+args.batch_size],dtype=torch.long,device=args.device) for c,v in meta.items()}
                    vm=model.meta_encoder(ix)
                    if args.kind=='concat':logits=model.classifier(torch.cat([token[:,0],vm],dim=1))
                    else:logits=model.head(token,vm)
                    probs.append(torch.softmax(logits,dim=1).cpu().numpy())
            return np.concatenate(probs)
        p=predict(metadata);truth=test.label.map(label2id).to_numpy();pred=p.argmax(1)
        acc=float((truth==pred).mean());expected=float(reference.loc[reference.fold==fold_no,'accuracy'].iloc[0])
        # Mismatch must be investigated before any probabilities are accepted as historical OOF.
        if not np.isclose(acc,expected,atol=1e-10):
            write_json(out/'RECOVERY_FAILED.json',dict(fold=fold_no,recovered_accuracy=acc,expected_accuracy=expected,
                reason='Checkpoint, config, metadata order, preprocessing, or numerical-version mismatch. No oof_predictions.csv written.'))
            raise ValueError('Recovered accuracy does not match released fold; inspect RECOVERY_FAILED.json')
        for i,row in test.iterrows():
            predictions.append(dict(sample_id=row.sample_id,model=args.kind,fold=fold_no,
                               y_true=classes[truth[i]],y_pred=classes[pred[i]],**{f'p_{c}':float(p[i,c]) for c in range(5)}))
        base_loss=log_loss(truth,p,labels=np.arange(5))
        for repeat in range(args.permutations):
            for mode in ['joint_tuple','independent_columns']:
                order=rng.permutation(len(test))
                shuffled={c:v[order if mode=='joint_tuple' else rng.permutation(len(test))] for c,v in metadata.items()}
                q=predict(shuffled)
                changed=np.any(np.stack([shuffled[c]!=metadata[c] for c in metadata]),axis=0).mean()
                perturbations.append(dict(model=args.kind,fold=fold_no,repeat=repeat,condition=mode,n=len(test),
                    original_accuracy=acc,permuted_accuracy=float((q.argmax(1)==truth).mean()),
                    delta_accuracy=float((q.argmax(1)==truth).mean()-acc),original_log_loss=base_loss,
                    permuted_log_loss=log_loss(truth,q,labels=np.arange(5)),fraction_metadata_tuples_changed=changed))
        if args.save_features:
            feature_parts.append(hidden[:,0].numpy());feature_ids.extend(test.sample_id.tolist())
        runlog.append(dict(fold=fold_no,checkpoint_sha256=sha256(checkpoint),n_test=len(test),accuracy=acc,
                           removed_thop_counter_keys=counters,metadata_vocabulary=vocab.vocabs))
        del model,backbone,hidden
        if torch.cuda.is_available():torch.cuda.empty_cache()
        print(f'Validated {args.kind} fold {fold_no}: n={len(test)}, accuracy={acc:.6f}')
    pd.DataFrame(predictions).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(perturbations).to_csv(out/'inference_metadata_permutations.csv',index=False)
    if args.save_features:np.savez_compressed(out/'frozen_cls_features.npz',sample_id=np.asarray(feature_ids,dtype=str),features=np.concatenate(feature_parts))
    write_json(out/'recovery_provenance.json',dict(notebook=str(notebook),notebook_sha256=sha256(notebook),
        metadata_sha256=sha256(args.metadata),configuration=cfg.to_dict(),gate_scale=args.gate_scale,
        image_path_mode='unique basename lookup in explicitly flat image directory' if args.flat_images else 'metadata path relative to data root',
        torch_version=torch.__version__,folds=runlog,backbone_sha256=backbone_hash,
        split_status='Reconstructed using author-confirmed original row order, seed 42 and original stratified splitter; marginal count fingerprint verified',
        limitations='Accuracy agreement and count fingerprints do not prove historic image identity without original manifests; retain this limitation. Inference permutations measure reliance, not complementary causal information or retrained-model utility.'))


if __name__=='__main__':main()
