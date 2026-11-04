"""Prevent silently substituting the released model for an ABR-trained checkpoint."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kev_initialization import resolve_initialization


class InitializationTests(unittest.TestCase):
    def test_abr_default_explicit_release_and_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta={'base':'fixture','base_revision':'fixed','lora':16,'head_dim':256,'weights_dtype':'fp32'}
            config={'default_source':'abr_baseline','sources':{}}
            for name in ['abr_baseline','released']:
                ck=root/name;ck.mkdir();(ck/'head.pt').write_bytes(name.encode())
                config['sources'][name]={**meta,'checkpoint_path':name,
                    'files_sha256':{'head.pt':hashlib.sha256(name.encode()).hexdigest()},'selection_note':'fixture'}
            reader=lambda path:dict(meta)
            args=(config,root,meta)
            self.assertEqual(resolve_initialization(*args,metadata_reader=reader)['source_name'],'abr_baseline')
            self.assertEqual(resolve_initialization(*args,source='released',metadata_reader=reader)['source_name'],'released')
            (root/'abr_baseline/head.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Pinned initialization changed'):
                resolve_initialization(*args,metadata_reader=reader)

    def test_base_mismatch_does_not_trigger_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'head.pt').write_bytes(b'x')
            meta={'base':'one','base_revision':'fixed','lora':16,'head_dim':256,'weights_dtype':'fp32'}
            config={'default_source':'abr_baseline','sources':{'abr_baseline':{**meta,'checkpoint_path':'.',
               'files_sha256':{'head.pt':hashlib.sha256(b'x').hexdigest()},'selection_note':'fixture'}}}
            with self.assertRaisesRegex(ValueError,'cached base differ'):
                resolve_initialization(config,root,{**meta,'base':'other'},metadata_reader=lambda p:meta)


if __name__=='__main__':unittest.main()
