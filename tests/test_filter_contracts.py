import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'chapters/03_compliance'
spec = importlib.util.spec_from_file_location('filter_core', FOLDER/'filter_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

class FilterContracts(unittest.TestCase):
    def test_risk_score_is_not_the_top_safe_score(self):
        result = core.risk_result([{'label':'normal','score':.99},
                                   {'label':'nsfw','score':.01}], 'NSFW', .5)
        self.assertEqual(result['status'], 'clear')
        self.assertEqual(result['score'], .01)
        self.assertEqual(core.risk_result([{'label':'NSFW','score':.5}], 'nsfw', .5)['status'], 'blocked')

    def test_missing_or_invalid_risk_score_is_rejected(self):
        for scores in [[{'label':'other','score':.9}], [{'label':'NSFW','score':float('nan')}]]:
            with self.assertRaises(ValueError):
                core.risk_result(scores, 'NSFW', .5)

    def test_keyword_normalization_and_context_false_positive(self):
        self.assertEqual(core.keyword_check('ＮＵＤＥ portrait', ['nude'])['status'], 'blocked')
        self.assertEqual(core.keyword_check('nude-colored lipstick', ['nude'])['status'], 'blocked')
        self.assertEqual(core.keyword_check('a red cup', ['nude'])['status'], 'clear')

    def test_audit_still_scores_text_after_keyword_match(self):
        calls = []
        def loader(kind, config, device):
            def predict(value, **kwargs):
                calls.append(kind)
                return [{'label':'NSFW','score':.1}]
            return predict
        with tempfile.TemporaryDirectory() as temp:
            result = core.compare(FOLDER, temp, loader=loader)
            self.assertEqual(calls.count('text'), 8)
            self.assertEqual(calls.count('image'), 2)
            self.assertEqual(result['failures'], 0)
            self.assertFalse(result['image_positive_control_present'])

    def test_model_loading_error_remains_unavailable(self):
        def loader(kind, config, device):
            if kind == 'text':
                raise RuntimeError('test load failure')
            return lambda value, **kwargs: [{'label':'nsfw','score':.01}]
        with tempfile.TemporaryDirectory() as temp:
            result = core.compare(FOLDER, temp, loader=loader)
            self.assertEqual(result['text_scored'], 0)
            self.assertEqual(result['image_scored'], 2)
            self.assertEqual(result['failures'], 1)
            import csv
            with open(pathlib.Path(temp)/'text_results.csv', encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(x['text_status']=='unavailable' for x in rows))

    def test_keyword_only_never_loads_models(self):
        def loader(*args):
            raise AssertionError('must not load')
        with tempfile.TemporaryDirectory() as temp:
            result = core.compare(FOLDER, temp, keyword_only=True, loader=loader)
            self.assertEqual(result['failures'], 0)
            self.assertEqual(result['text_scored'], 0)
            self.assertEqual(result['image_scored'], 0)

if __name__ == '__main__':
    unittest.main()
