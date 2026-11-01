import json
from pathlib import Path
import sys,tarfile,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deploy_stage_checkout import inventory,verify_archive,write_new

class StageDeploymentTests(unittest.TestCase):
    def test_archive_preserves_existing_content_without_restoring_missing_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); code=root/'code'; code.mkdir()
            (code/'present.txt').write_text('original')
            (code/'.git').mkdir(); (code/'.git/HEAD').write_text('ref: refs/heads/original\n')
            manifest=inventory(code); archive=root/'old.tar.gz'
            with tarfile.open(archive,'x:gz',dereference=False) as h: h.add(code,arcname='code')
            verify_archive(archive,manifest)
            self.assertEqual(inventory(code),manifest)
            self.assertFalse((code/'historical_missing.txt').exists())
            self.assertNotIn('historical_missing.txt',manifest)

    def test_archive_rejects_wrong_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); code=root/'code'; code.mkdir(); (code/'x').write_text('data')
            archive=root/'old.tar.gz'
            with tarfile.open(archive,'x:gz') as h: h.add(code,arcname='code')
            expected=inventory(code); expected['x']['sha256']='0'*64
            with self.assertRaises(ValueError): verify_archive(archive,expected)

    def test_receipt_is_write_once_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json'
            write_new(path,{'source':'fixed'}); write_new(path,{'source':'fixed'})
            with self.assertRaises(RuntimeError): write_new(path,{'source':'changed'})
            self.assertEqual(json.loads(path.read_text()),{'source':'fixed'})

if __name__=='__main__': unittest.main()
