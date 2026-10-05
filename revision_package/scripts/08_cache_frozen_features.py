"""One inference pass to cache CLS features from an existing LOCAL pretrained DINOv3.

Use if original disease checkpoints are unavailable. Does not train or download.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from audit_common import new_output,write_json,sha256,resolve_image,build_flat_image_index


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-path',required=True,help='Local pretrained HF snapshot, independent of these disease labels')
    ap.add_argument('--metadata',required=True);ap.add_argument('--data-root',required=True)
    ap.add_argument('--strip-prefix',default='');ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--flat-images',action='store_true',
                    help='Match metadata paths to a flat image directory by unique basename')
    ap.add_argument('--device',default='cuda');ap.add_argument('--out',required=True)
    args=ap.parse_args()
    import torch
    from transformers import AutoModel
    from torchvision import transforms
    from PIL import Image
    if args.batch_size<1:raise ValueError('Positive batch size required')
    d=pd.read_csv(args.metadata)
    flat_index=build_flat_image_index(args.data_root) if args.flat_images else None
    resolved=[resolve_image(r,args.data_root,args.strip_prefix,flat_index=flat_index) for _,r in d.iterrows()]
    ids=[sid for p,sid in resolved]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate image IDs')
    if 'sample_id' in d and d.sample_id.astype(str).tolist()!=ids:raise ValueError('sample_id differs from canonical relative path')
    model=AutoModel.from_pretrained(args.model_path,local_files_only=True).to(args.device).eval()
    for p in model.parameters():p.requires_grad=False
    if model.config.hidden_size!=768:raise ValueError('Expected ViT-B hidden size 768')
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                           transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    arrays=[]
    with torch.inference_mode():
        for start in range(0,len(d),args.batch_size):
            batch=[]
            for path,sid in resolved[start:start+args.batch_size]:
                with Image.open(path) as im:batch.append(tfm(im.convert('RGB')))
            arrays.append(model(pixel_values=torch.stack(batch).to(args.device)).last_hidden_state[:,0].cpu().numpy())
    if not arrays:raise ValueError('Empty metadata')
    out=new_output(args.out)
    np.savez_compressed(out/'frozen_cls_features.npz',sample_id=np.asarray(ids,dtype=str),features=np.concatenate(arrays))
    local_model=Path(args.model_path)
    weight_files=sorted([p for p in local_model.glob('*') if p.is_file() and p.suffix in ['.safetensors','.bin','.json']]) if local_model.exists() else []
    write_json(out/'feature_provenance.json',dict(model_path=args.model_path,model_files={p.name:sha256(p) for p in weight_files},
        model_config=model.config.to_dict(),metadata_sha256=sha256(args.metadata),n=len(d),preprocessing='RGB, bilinear resize to 224x224, ImageNet normalization; no augmentation',
        image_path_mode='unique basename lookup in explicitly flat image directory' if args.flat_images else 'metadata path relative to data root',
        scope='NEW fixed-feature diagnostic experiment. Author must verify pretrained model independence from disease evaluation labels.',torch_version=torch.__version__))


if __name__=='__main__':main()
