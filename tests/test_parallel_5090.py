"""Task partition and four-device gallery regression checks."""
import unittest
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from appendix_flexible import offline_plan
import appendix_parallel_figures as gallery

class ParallelTests(unittest.TestCase):
    def test_exact_old_curve_partition(self):
        spec=json.loads(Path('configs/appendix-parallel-5090.json').read_text())
        sets=[]
        for name in spec['tasks']:
            jobs=offline_plan(spec,[name])['jobs']
            self.assertEqual(len(jobs),10 if name.startswith('E') else 7)
            sets.append({(j['family'],j['name']) for j in jobs})
        self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
        old=offline_plan(spec,['A','B'])['jobs']
        expected={(j['family'],j['name']) for j in old if j['panel']=='curve'}
        self.assertEqual(set.union(*sets),expected)
        self.assertEqual(offline_plan(spec,['core'])['target_slots'],79*512)

    def test_gallery_four_devices_and_h100_policy(self):
        with TemporaryDirectory() as tmp:
            jobs=[dict(name='job'+str(i),blocked=None) for i in range(3)]
            core={'interpreters':{'render':sys.executable}}
            plan={'jobs':jobs,'family':'sd15'}
            with patch.object(gallery,'figure_plan',return_value=(plan,core)), \
                 patch('appendix_jobs.gpu_tokens',return_value=['0','1','2','3']), \
                 patch.object(gallery,'checked_runtime',return_value={'gpu':'NVIDIA GeForce RTX 5090'}), \
                 patch.object(gallery,'source_index'),patch.object(gallery,'results',return_value=None), \
                 patch.object(gallery,'export',return_value={}),patch.object(gallery,'run_workers') as workers:
                gallery.run(Path(tmp),'sd15')
                self.assertEqual([len(c.args[0]) for c in workers.call_args_list],[4,2])
                self.assertEqual(workers.call_args_list[0].args[2]['MVPO_FIGURE_BATCH'],'16')
                with self.assertRaisesRegex(ValueError,'H100'):
                    gallery.run(Path(tmp),'zimage')

if __name__=='__main__':unittest.main()
