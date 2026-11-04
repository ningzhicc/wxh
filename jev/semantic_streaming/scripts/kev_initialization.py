"""Select an explicitly pinned local initialization; never fall back to another model."""
import hashlib
from pathlib import Path


def resolve_initialization(config, workspace_root, deployment, source=None, metadata_reader=None):
    name=source or config['default_source']
    if name not in config['sources']:
        raise ValueError('Unknown initialization source: '+name)
    selected=config['sources'][name]
    root=Path(workspace_root).resolve()
    checkpoint=(root/selected['checkpoint_path']).resolve()
    if not checkpoint.is_relative_to(root):
        raise ValueError('Initialization checkpoint escapes workspace')
    for filename,expected in selected['files_sha256'].items():
        if hashlib.sha256((checkpoint/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('Pinned initialization changed: '+filename)
    if metadata_reader is None:
        import torch
        metadata_reader=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    meta=metadata_reader(checkpoint/'head.pt')
    for field in ('base','base_revision','lora','head_dim','weights_dtype'):
        if meta.get(field)!=selected[field]:
            raise ValueError('Checkpoint metadata differs from pinned initialization: '+field)
    for field in ('base','base_revision'):
        if meta[field]!=deployment[field]:
            raise ValueError('Initialization and cached base differ: '+field)
    if (meta['lora']!=16 or meta['head_dim']!=256 or meta['weights_dtype']!='fp32'
            or meta.get('option_isolation',False) or meta.get('special_embeddings',False)):
        raise ValueError('Initialization architecture does not match this training recipe')
    return {'source_name':name,'checkpoint_path':str(checkpoint),
            **{k:selected[k] for k in ('files_sha256','base','base_revision','lora','head_dim','weights_dtype','selection_note')}}
