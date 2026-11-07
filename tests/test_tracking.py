"""Real offline W&B records across training, evaluation and resume segments."""
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

from self_correct_dpo.runtime.tracking import ExperimentTracker

spec = importlib.util.spec_from_file_location('sync_flow_wandb',
    Path(__file__).resolve().parents[1]/'scripts/sync_flow_wandb.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class OfflineTrackingTest(unittest.TestCase):
    def test_unfinished_older_history_blocks_newer_and_overlap_is_refused(self):
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'training': {'wandb': {'project': 'mvpo-flow', 'mode': 'offline'}}}
            (root/'training.json').write_text(json.dumps({'run_id': 'd'*64, 'config': config}))
            def segment(name, steps, finished):
                path = root/'wandb'/('offline-run-'+name)/'run-example.wandb'
                path.parent.mkdir(parents=True);store = DataStore();store.open_for_write(str(path))
                header = pb.Record();header.header.SetInParent();store.write(header)
                store.write(pb.Record(run=pb.RunRecord(run_id='d'*32, project='mvpo-flow')))
                for step in steps:
                    history = pb.HistoryRecord()
                    history.item.add(key='_step', value_json=str(step))
                    store.write(pb.Record(history=history))
                if finished:store.write(pb.Record(exit=pb.RunExitRecord()))
                store.close()
                return path
            segment('newer', [10, 12], True)
            active = segment('older', [5], False)
            self.assertEqual(sync.plan_sync(root, 'mvpo-flow')[0], [])
            active.unlink();active.parent.rmdir()
            segment('older-finished', [5], True)
            self.assertEqual([p['first'] for p in sync.plan_sync(root, 'mvpo-flow')[0]], [5, 10])
            segment('overlap', [11], True)
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                sync.plan_sync(root, 'mvpo-flow')

    def test_segments_share_identity_and_global_history_steps(self):
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
        config = {'evaluation': {'enabled': True, 'include_base': True},
                  'training': {'logging': 'wandb', 'wandb': {
            'project': 'mvpo-flow', 'mode': 'offline',
            'group': 'zimage-hpdv3-pair_views128-seed17',
            'name': 'zimage-stagewise-k2-C-test', 'tags': ['zimage', 'k2', 'stagewise'],
        }}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'training.json').write_text(json.dumps({'run_id': 'a'*64, 'config': config}))
            # The real pipeline creates the base evaluation after training.
            for evaluation, steps in ((False, [1, 2]), (True, [0]), (True, [2]), (False, [3])):
                tracker = ExperimentTracker(config, directory, 'a' * 64,
                    start_step=steps[0] - 1, resume=True, evaluation=evaluation)
                for step in steps:
                    tracker.add_scalar('evaluation/pickscore' if evaluation else 'training/loss', .5, step)
                    tracker.flush()
                tracker.close()
                if not evaluation and steps == [1, 2]:
                    plans, message = sync.plan_sync(root, 'mvpo-flow')
                    self.assertEqual(plans, [])
                    self.assertIn('step-0', message)
            histories, identities = [], []
            for path in Path(directory).glob('wandb/offline-run-*/run-*.wandb'):
                store = DataStore()
                store.open_for_scan(str(path))
                try:
                    while (data := store.scan_data()) is not None:
                        record = wandb_internal_pb2.Record()
                        record.ParseFromString(data)
                        if record.HasField('history'):
                            histories.append({item.key or '.'.join(item.nested_key): json.loads(item.value_json)
                                              for item in record.history.item})
                        if record.HasField('run'):
                            identities.append((record.run.run_id, record.run.project,
                                               record.run.run_group, record.run.display_name))
                finally:
                    store.close()
            self.assertEqual(len(identities), 4)
            self.assertEqual(set(identities), {('a' * 32, 'mvpo-flow',
                'zimage-hpdv3-pair_views128-seed17', 'zimage-stagewise-k2-C-test')})
            self.assertTrue(all('_step' in row for row in histories), histories)
            histories.sort(key=lambda row: row['_step'])
            self.assertEqual([row['_step'] for row in histories], [1, 2, 4, 5, 6])
            self.assertEqual([row['optimizer_step'] for row in histories], [0, 1, 2, 2, 3])
            self.assertIn('evaluation/pickscore', histories[3])
            files = list(Path(directory).glob('wandb/offline-run-*/run-*.wandb'))
            self.assertEqual(sorted(sync.history_start(path) for path in files), [1, 2, 5, 6])
            self.assertTrue(all(sync.inspect_segment(path)['finished'] for path in files))
            # A valid but still open segment and an empty just-created file must be skipped.
            active = root/'wandb/offline-run-active/run-active.wandb'
            active.parent.mkdir()
            writer = DataStore();writer.open_for_write(str(active))
            writer.write(wandb_internal_pb2.Record(run=wandb_internal_pb2.RunRecord(
                run_id='a'*32, project='mvpo-flow')))
            writer.close()
            plans, message = sync.plan_sync(root, 'mvpo-flow')
            self.assertEqual([p['first'] for p in plans], [1, 2, 5, 6])
            self.assertIn('1 unfinished', message)
            active.write_bytes(b'')
            self.assertFalse(sync.inspect_segment(active)['finished'])
            self.assertEqual(sync.sync_run(root, 'example-team', 'mvpo-flow', dry_run=True), 4)
            self.assertFalse((root/'.flow-wandb-sync.json').exists())
            commands = []
            def upload(command, check):
                commands.append(command)
                folder = Path(command[-1])
                next(folder.glob('run-*.wandb')).with_suffix('.wandb.synced').touch()
            sync.sync_run(root, 'example-team', 'mvpo-flow', runner=upload)
            self.assertEqual(len(commands), 4)
            self.assertTrue(all('--append' in command for command in commands))
            self.assertEqual(sync.sync_run(root, 'example-team', 'mvpo-flow', runner=upload), 0)
            with self.assertRaisesRegex(ValueError, 'different destination'):
                sync.sync_run(root, 'other-team', 'mvpo-flow', dry_run=True)

    def test_evaluation_images_are_in_same_offline_history_row(self):
        from PIL import Image
        from self_correct_dpo.runtime.tracking import log_evaluation
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
        config = {'training': {'logging': 'wandb', 'wandb': {
            'project': 'mvpo-flow', 'mode': 'offline', 'image_count': 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'training.json').write_text(json.dumps({'run_id': 'b'*64}))
            render = root/'evaluations/step-0000200/render'
            (render/'images').mkdir(parents=True)
            Image.new('RGB', (16, 16), 'red').save(render/'images/a.png')
            (render/'manifest.jsonl').write_text(json.dumps({'scene_id': 'same-scene', 'seed': 17,
                'prompt': 'red image', 'image': 'images/a.png'})+'\n')
            log_evaluation(config, root, 200, {'metrics': {'pickscore': {'mean': 22.5}}, 'images_retained': True})
            path = next(root.glob('wandb/offline-run-*/run-*.wandb'))
            store = DataStore();store.open_for_scan(str(path));rows = []
            try:
                while (data := store.scan_data()) is not None:
                    record = wandb_internal_pb2.Record();record.ParseFromString(data)
                    if record.HasField('history'):
                        rows.append({i.key or '.'.join(i.nested_key): json.loads(i.value_json)
                                     for i in record.history.item})
            finally:
                store.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]['_step'], rows[0]['optimizer_step']), (401, 200))
            self.assertEqual(rows[0]['evaluation/pickscore'], 22.5)
            self.assertEqual(rows[0]['evaluation/images.count'], 1)
            self.assertEqual(len(list(path.parent.glob('files/media/images/**/*.png'))), 1)


if __name__ == '__main__':
    unittest.main()
