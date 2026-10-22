"""Regression: SLD may return scalar flags while standard SD returns one-element lists."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/a800'))
from common import normalize_output_flag
class OutputCheckerFlags(unittest.TestCase):
 def test_scalar_and_vector_flags_keep_the_same_meaning(self):
  for value,expected in [(False,False),(True,True),([False],False),([True],True),((False,),False)]:
   with self.subTest(value=value):self.assertIs(normalize_output_flag(value),expected)
 def test_unavailable_is_not_clear(self):self.assertIsNone(normalize_output_flag(None))
 def test_wrong_batch_and_unknown_scalar_are_rejected(self):
  for value in [[],[False,True],0.0,'unexpected']:
   with self.subTest(value=value):
    with self.assertRaises(ValueError):normalize_output_flag(value)
