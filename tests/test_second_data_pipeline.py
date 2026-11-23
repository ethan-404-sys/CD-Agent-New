"""CPU-only tests of the SECOND plan/receipt/export boundaries."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cd_agent import verify
from experiments.verifier_diag import prepare_second_masks as second
from experiments.verifier_diag.scannet_adapter import canonical_semantics


def truth():
    t1 = np.zeros((32, 32), dtype=np.uint8)
    t2 = t1.copy()
    t1[3:13, 3:13], t2[3:13, 3:13] = 1, 3
    t1[18:28, 18:28], t2[18:28, 18:28] = 3, 5
    return t1 != t2, t1, t2


class SecondDataTest(unittest.TestCase):
    def test_frozen_official_splits(self):
        split = second.frozen_splits()
        self.assertEqual(verify.check_split_isolation(split),
                         dict(ok=True, counts={'train': 2568, 'val': 400, 'test': 1694}, overlaps={}))
        official = second.official_names('train', 2968)
        random.Random(20261122).shuffle(official)
        self.assertEqual(split['val'], official[:400])
        self.assertEqual(split['train'], official[400:])
        self.assertEqual(split['test'], second.official_names('test', 1694))
        with self.assertRaisesRegex(ValueError, 'split leakage'):
            verify.check_split_isolation({'train': split['train'], 'val': split['val'] + split['train'][:1]})

    def test_scannet_canonical_mapping(self):
        np.testing.assert_array_equal(canonical_semantics(np.arange(7)), [0, 4, 2, 1, 3, 5, 6])
        with self.assertRaises(ValueError):
            canonical_semantics(np.array([-1, 7]))

    def test_task_compilation(self):
        change, t1, t2 = truth()
        tasks = second.task_catalog()
        self.assertEqual(len(tasks), 40)
        self.assertNotIn('transition_1_2', {task['id'] for task in tasks})
        self.assertNotIn('transition_2_1', {task['id'] for task in tasks})
        self.assertEqual({task['task_params']['class_id'] for task in tasks
                          if task['task_type'] != 'transition'}, {3, 4, 5, 6})
        targets = second.eligible_targets(change, t1, t2, tasks)
        self.assertEqual({key for key in targets if key.startswith('transition')},
                         {'transition_1_3', 'transition_3_5'})
        np.testing.assert_array_equal(targets['related_3'], change)
        np.testing.assert_array_equal(targets['into_3'], targets['transition_1_3'])
        np.testing.assert_array_equal(targets['outof_3'], targets['transition_3_5'])
        self.assertFalse(targets['outof_5'].any())

    def test_all6_task_catalog_and_targets(self):
        t1 = np.zeros((32, 32), dtype=np.uint8)
        t2 = t1.copy()
        t1[3:13, 3:13], t2[3:13, 3:13] = 1, 2
        t1[18:28, 18:28], t2[18:28, 18:28] = 2, 5
        change = t1 != t2
        tasks = second.task_catalog('all6')
        self.assertEqual(len(tasks), 48)
        ids = {task['id'] for task in tasks}
        self.assertIn('related_1', ids)
        self.assertIn('related_2', ids)
        self.assertIn('transition_1_2', ids)
        self.assertIn('transition_2_1', ids)
        self.assertEqual({task['task_params']['class_id'] for task in tasks
                          if task['task_type'] != 'transition'}, {1, 2, 3, 4, 5, 6})
        targets = second.eligible_targets(change, t1, t2, tasks)
        self.assertEqual({key for key in targets if key.startswith('transition')},
                         {'transition_1_2', 'transition_2_5'})
        np.testing.assert_array_equal(targets['outof_1'], targets['transition_1_2'])
        np.testing.assert_array_equal(targets['into_2'], targets['transition_1_2'])
        names = second.frozen_splits()['val']
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / 'dataset'
            out = Path(tmp) / 'run6'
            for folder in ('img1', 'img2'):
                (dataset / folder).mkdir(parents=True)
                Image.new('RGB', (32, 32)).save(dataset / folder / names[0])
            plan = second.prepare(out, 'val', dataset=dataset, limit=1, class_mode='all6')
            self.assertEqual(plan['class_mode'], 'all6')
            self.assertEqual(len(plan['tasks']), 48)
            self.assertEqual(second.load_plan(out)['class_mode'], 'all6')

    def test_partial_or_fragmented_transition_rejected(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:8, 2:8] = True
        mask[2:8, 9:15] = True
        proxy = mask.copy()
        proxy[4, 8] = True
        self.assertTrue(second.complete_transition(proxy, proxy))
        self.assertFalse(second.complete_transition(mask, proxy))
        mask[2:8, 9:15] = False
        self.assertFalse(second.complete_transition(mask, proxy))

    def test_clean_labels_and_exact_band_boundaries(self):
        raw, _, _ = truth()
        raw[1, 30] = True
        pred = np.zeros_like(raw)
        pred[3:13, 3:13] = True
        pred[20:25, 3:8] = True
        labels = second.clean_diagnostic(pred, raw)
        self.assertTrue(labels['ignore'][1, 30])
        self.assertFalse(labels['gt'][1, 30])
        self.assertTrue(labels['ignore'][3, 3])
        self.assertTrue(labels['fp'][20:25, 3:8].all())
        self.assertTrue(labels['fn'][18:28, 18:28].all())
        self.assertEqual(second.label_metrics(pred, labels)['fn_area'], 100)
        with self.assertRaisesRegex(ValueError, 'invariant'):
            verify.check_diagnostic_labels(pred, labels['gt'], labels['fn'], labels['fp'], labels['ignore'])
        self.assertEqual([second.difficulty(x) for x in (0, .3, .7, .9)], list(second.BANDS))
        self.assertEqual(second.difficulty(1), second.BANDS[-1])

    def test_prepare_infer_resume_then_export_without_gt_in_inference(self):
        change, t1, t2 = truth()
        names = second.frozen_splits()['train']
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / 'dataset'
            out = Path(tmp) / 'run'
            for folder in ('img1', 'img2'):
                (dataset / folder).mkdir(parents=True)
                for name in (names[0], names[1]):
                    Image.new('RGB', (32, 32)).save(dataset / folder / name)
            plan = second.prepare(out, 'train', dataset=dataset, limit=1, shard=1, num_shards=2)
            self.assertEqual(plan['pairs'][0]['id'], names[1])
            self.assertEqual(plan, second.prepare(out, 'train', dataset=dataset, limit=1, shard=1, num_shards=2))
            with self.assertRaisesRegex(ValueError, 'different request'):
                second.prepare(out, 'train', dataset=dataset, limit=1)
            calls = []

            def predict(image_t1, image_t2, tasks):
                self.assertEqual({Path(image_t1).parent.name, Path(image_t2).parent.name}, {'img1', 'img2'})
                self.assertEqual(tasks, second.task_catalog())
                calls.append((image_t1, image_t2))
                return [second.compile_target(change, t1, t2, task) for task in tasks]

            with patch.dict('os.environ', {'SLURM_JOB_ID': ''}):
                with self.assertRaisesRegex(RuntimeError, 'SLURM_JOB_ID'):
                    second.infer(out, 'scannet')

            with (patch.dict('os.environ', {'SLURM_JOB_ID': '123'}),
                  patch.object(second, 'load_truth', side_effect=AssertionError('GT read during inference')),
                  patch.object(second, 'model_settings', return_value={'checkpoint': 'stub'}),
                  patch.object(second, 'build_adapter', return_value=predict) as build):
                for method in second.METHODS:
                    second.infer(out, method)
                    second.infer(out, method)
                self.assertEqual(build.call_count, len(second.METHODS))
            self.assertEqual(len(calls), len(second.METHODS))

            # No ground-truth files existed until both preparation and inference were done.
            for folder, value in (('slabel1', t1), ('slabel2', t2),
                                  ('label_unch0chi1', change.astype(np.uint8))):
                (dataset / folder).mkdir()
                Image.fromarray(value).save(dataset / folder / names[1])
            metrics = second.export(out)
            self.assertEqual(metrics['pairs'], 1)
            self.assertEqual(metrics['record_types']['counterfactual'], len(second.METHODS))
            self.assertEqual(metrics['record_types']['anchor'], 1)
            self.assertEqual(sum(sum(hist.values()) for hist in metrics['by_generator'].values()),
                             metrics['record_types']['generated'])
            rows = [json.loads(line) for line in (out / 'manifest.jsonl').read_text().splitlines()]
            self.assertTrue(verify.verify_run_dir(out, expected_count=len(rows))['ok'])
            by_id = {row['id']: row for row in rows}
            for row in rows:
                if row['is_counterfactual']:
                    base = by_id[row['base_task_id']]
                    for key in ('image_t1', 'image_t2', 'init_mask', 'counterfactual_group_id'):
                        self.assertEqual(row[key], base[key])
                    self.assertNotEqual(row['instruction'], base['instruction'])
                    inputs = []
                    for item in (base, row):
                        inputs.append(dict(gt=second.read_pixels(Path(item['gt_mask'])),
                                           ignore=second.read_pixels(Path(item['labels']['ignore_mask']))))
                    self.assertEqual(row['effective_gt_diff_area'], second.effective_difference(*inputs))
                    self.assertGreater(row['effective_gt_diff_area'], 0)
                if row['is_anchor']:
                    self.assertEqual(row['init_mask'], row['gt_mask'])
                    self.assertEqual(row['metrics']['fp_area'] + row['metrics']['fn_area'], 0)
            self.assertIn('SECOND initial masks', (out / 'viz' / 'index.html').read_text())
            selected = json.loads((out / 'viz' / 'selection.json').read_text())
            self.assertTrue(selected)
            self.assertTrue(list((out / 'viz' / 'assets').glob('*.png')))


if __name__ == '__main__':
    unittest.main()
