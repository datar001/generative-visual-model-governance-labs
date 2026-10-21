import importlib.util
import pathlib
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cup_search', ROOT/'chapters/03_compliance/pseudoword_core.py')
mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class PseudowordContracts(unittest.TestCase):
    def test_strategies_use_equal_unique_query_budgets(self):
        starts = []
        for mode in ('random','improve'):
            calls = []
            def score(prompt):
                calls.append(prompt)
                return sum(ord(x) for x in prompt)/1000
            trace = mod.search(['baflo','vemri','nuvak'], 3, 2026, 16, mode, score)
            self.assertEqual(len(calls),16)
            self.assertEqual(len(set(calls)),16)
            self.assertEqual(trace[-1]['query'],16)
            starts.append(trace[0]['candidate'])
        self.assertEqual(starts[0],starts[1])

    def test_larger_budget_preserves_the_same_search_prefix(self):
        for mode in ('random','improve'):
            f = lambda s: sum(ord(c) for c in s)/1000
            small = mod.search(['baflo','vemri','nuvak'],3,2,8,mode,f)
            large = mod.search(['baflo','vemri','nuvak'],3,2,16,mode,f)
            self.assertEqual([x['candidate'] for x in small], [x['candidate'] for x in large[:8]])

    def test_running_best_is_monotone_and_matches_observed_scores(self):
        trace = mod.search(['baflo','vemri','nuvak'],3,3,16,'improve',lambda s: sum(ord(c) for c in s)/1000)
        observed = []
        for row in trace:
            observed.append(row['score'])
            self.assertEqual(row['best_score'],max(observed))

    def test_invalid_scores_and_excessive_budget_are_not_ranked(self):
        with self.assertRaises(ValueError):
            mod.search(['a','b'],2,0,5,'random',lambda s:0)
        with self.assertRaises(ValueError):
            mod.search(['a','b'],2,0,2,'random',lambda s:float('nan'))

    def test_classroom_gate_is_a_benign_literal_rule(self):
        self.assertFalse(mod.classroom_gate('a red cup',['red','cup']))
        self.assertTrue(mod.classroom_gate('baflo nuvak',['red','cup']))
if __name__ == '__main__':unittest.main()
