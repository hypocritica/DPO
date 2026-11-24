"""Check seed selection against the real L design and coordinator dispatch."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import appendix_flexible as flex
import run_lossavg_k4_seed as seed_task


class LossAvgSeedTests(unittest.TestCase):
    def test_exact_disjoint_l_jobs(self):
        spec = flex.read_json('configs/appendix-parallel-5090.json')
        jobs = flex.offline_plan(spec, ['L', 'sd35-lossavg_k4'])['jobs']
        for job in jobs:
            job['shards'] = 4
        selected = [seed_task.select_job({'jobs': jobs}, seed)[0][1] for seed in (17, 23, 42)]
        self.assertEqual({j['name'] for j in selected},
                         {f'lossavg_k4-1600-s{s}' for s in (17, 23, 42)})
        self.assertEqual(sum(spec[j['panel'] + '_count'] for j in selected), 4896)
        with self.assertRaises(ValueError):
            seed_task.select_job({'jobs': jobs + [copy.deepcopy(selected[0])]}, 17)

    def test_four_card_launch_keeps_original_worker_and_plan(self):
        spec = flex.read_json('configs/appendix-parallel-5090.json')
        job = dict(name='lossavg_k4-1600-s23', old_pack='L', method='lossavg_k4',
                   step=1600, panel='full', seed=23, shards=4)
        plan = dict(spec=spec, jobs=[job], interpreters={'render': '/test/python'})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(flex, 'load_resources', return_value={'MVPO_PARALLEL_ROOT': tmp}), \
             patch.object(flex, 'load_plan', return_value=plan), \
             patch.object(flex, 'wanted_rows', return_value=[None] * 408), \
             patch.object(flex, 'gpu_tokens', return_value=['4', '5', '6', '7']) as devices, \
             patch.object(flex, 'launch') as launch:
            seed_task.main(['generate', '--seed', '23', '--output', tmp])
            args, plans, selected = launch.call_args.args
            self.assertEqual(args.gpus, '4,5,6,7')
            self.assertEqual(selected, [('sd35_medium', job)])
            self.assertIs(plans['sd35_medium'], plan)
            self.assertIsNone(args.batch_size)
            devices.return_value = ['0', '1']
            with self.assertRaises(SystemExit):
                seed_task.main(['generate', '--seed', '23', '--output', tmp])
            self.assertEqual(launch.call_count, 1)


if __name__ == '__main__':
    unittest.main()
