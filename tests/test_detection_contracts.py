import importlib.util, os, pathlib, tempfile, unittest
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parents[1]/'chapters/04_detection/cnnspot_support.py'
spec=importlib.util.spec_from_file_location('support',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class WeightTrust(unittest.TestCase):
    def test_missing_weights_stop_without_download(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'CNN_SPOT_WEIGHTS':str(pathlib.Path(d)/'absent.pth')}):
            with self.assertRaises(FileNotFoundError):m.load_weights()
    def test_wrong_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'bad.pth';p.write_bytes(b'not the pinned checkpoint')
            with patch.dict(os.environ,{'CNN_SPOT_WEIGHTS':str(p)}):
                with self.assertRaises(ValueError):m.load_weights()
if __name__=='__main__':unittest.main()
