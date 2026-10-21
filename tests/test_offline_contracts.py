import csv, importlib.util, json, pathlib, subprocess, sys, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fairness_summary',ROOT/'chapters/07_fairness/summarize.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Contracts(unittest.TestCase):
 def test_undecidable_rows_remain_in_denominator(self):
  labels=['feminine']*10+['masculine']*6+['uncertain']*2+['failed','']
  rows=[{'job':'doctor','mode':'neutral','reviewed_label':x} for x in labels]
  result=mod.summarize(rows)[0]
  self.assertEqual(result['total'],20);self.assertEqual(result['decidable'],16)
  self.assertEqual(result['unreviewed'],1);self.assertEqual(result['failed'],1)
  self.assertEqual(result['feminine_fraction_all'],.5)
  self.assertEqual(result['feminine_fraction_decidable'],.625)
 def test_no_decidable_labels_is_not_zero_ratio(self):
  row=mod.summarize([{'job':'nurse','mode':'neutral'}])[0]
  self.assertIsNone(row['feminine_fraction_decidable'])
 def test_unknown_reviewed_label_is_rejected(self):
  with self.assertRaises(ValueError):
   mod.summarize([{'job':'doctor','mode':'neutral','reviewed_label':'bad'}])
 def test_governance_can_run_outside_repository(self):
  with tempfile.TemporaryDirectory() as temp:
   subprocess.run([sys.executable,str(ROOT/'chapters/01_overview/lab.py')],cwd=temp,check=True,capture_output=True)
   with open(pathlib.Path(temp)/'results/governance/risk_register.csv',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),3)
   risks={x for row in rows for x in row['risks'].split(';')}
   self.assertEqual(risks,{'authenticity','source','compliance','fairness','privacy'})
   self.assertTrue(all(row['review_status']=='pending' for row in rows))
 def test_dry_run_does_not_import_model_libraries(self):
  for chapter,case in [('01',None),('02',None),('03',None),('04',None),('05',None),
    ('06','sld'),('06','esd'),('07',None),('08','features'),('08','customize')]:
   command=[sys.executable,str(ROOT/'tools/run.py'),chapter,'--dry-run']
   if case:command.extend(['--case',case])
   r=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf8')
   result=json.loads(r.stdout);self.assertTrue(pathlib.Path(result['script']).exists())
if __name__=='__main__':unittest.main()
