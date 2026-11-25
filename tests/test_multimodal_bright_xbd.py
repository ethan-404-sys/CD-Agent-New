"""CPU-only tests for sampling, online isolation and offline reporting."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from cd_agent.planner import Planner
from experiments.multimodal_probe import bright_xbd_prepare as prep
from experiments.multimodal_probe import bright_xbd_run as run


class DamageProbeTest(unittest.TestCase):
    def test_binary_class_mapping(self):
        labels = np.array([[0, 1, 2, 3, 4, 5, 255]], dtype=np.uint8)
        self.assertEqual(prep.damage_mask(labels, 'bright').tolist(), [[False, False, True, True, False, False, False]])
        self.assertEqual(prep.damage_mask(labels, 'xbd').tolist(), [[False, False, True, True, True, False, False]])

    def test_sampling_event_balance_and_negatives(self):
        groups = {event: [prep.Source(f'{event}-{i}', event, None, None, (event, i)) for i in range(5)]
                  for event in ('a', 'b', 'c')}

        def labels(path):
            return np.array([[0, 1 if path[1] < 2 else 3]], dtype=np.uint8)

        with patch.object(prep, 'read_labels', side_effect=labels):
            sample = prep.select(groups, 'bright', count=9, negatives=2, seed=7)
            self.assertEqual(sample, prep.select(groups, 'bright', count=9, negatives=2, seed=7))
        self.assertEqual({event: sum(row[0].event == event for row in sample) for event in groups},
                         dict.fromkeys(groups, 3))
        self.assertEqual(len({src.event for src, damage, _ in sample if not damage}), 2)
        self.assertEqual(sum(not damage and intact > 0 for _, damage, intact in sample), 2)
        self.assertGreater(sample[0][1], 0)

    def test_sampling_positive_only_if_no_bright_negatives(self):
        groups = {'a': [prep.Source(str(i), 'a', None, None, i) for i in range(3)]}
        with patch.object(prep, 'read_labels', return_value=np.array([[3]], dtype=np.uint8)):
            sample = prep.select(groups, 'bright', count=3, negatives=2, seed=7)
        self.assertEqual(len(sample), 3)
        self.assertTrue(all(row[1] for row in sample))

    def test_routing_counts_t2_even_on_failure(self):
        calls = [
            dict(tool='segment_text', args=dict(time='t1'), ok=True),
            dict(tool='segment_text', args=dict(time='t2'), ok=False),
            dict(tool='segment_text', args=None, ok=False),
            dict(tool='python', args={}, ok=False),
            dict(tool='ask', args={}, ok=True),
            dict(tool='ask', args={}, status='skipped_due_to_submit'),
        ]
        self.assertEqual(run.routing([dict(calls=calls)], dict(planning_rounds=4, cost_cny=.25)),
                         dict(t1_optical_seg_calls=1, t2_optical_seg_calls=1, python_calls=1,
                              ask_calls=1, rounds=4, cost_cny=.25))

    def test_score_aggregation_distinguishes_micro_and_macro(self):
        rows = [dict(**run.scores(*values), event='event', failed=False, submitted_by='model',
                     routing=dict.fromkeys(run.ROUTING, 0)) for values in [(0, 0, 0), (2, 0, 2)]]
        summary = run.aggregate(rows)
        self.assertAlmostEqual(summary['micro']['f1'], 2 / 3)
        self.assertAlmostEqual(summary['macro']['f1'], 5 / 6)
        self.assertEqual(summary['macro']['precision'], .5)
        self.assertEqual(summary['micro']['recall'], .5)

    def test_mock_online_session_offline_metrics_and_evidence(self):
        size = (8, 8)
        counts = {'bright': 3, 'xbd': 1}
        with TemporaryDirectory() as tmp, \
                patch.multiple(prep, SIZE=size, COUNTS=counts, EVENT_COUNTS={'bright': 1, 'xbd': 1}), \
                patch.multiple(run, SIZE=size, COUNTS=counts):
            root = Path(tmp)
            source = root / 'datasets'
            bright = source / 'BRIGHT'
            val = bright / 'dfc25_track2_trainval' / 'dfc25_track2_trainval' / 'val'
            bright_labels = bright / 'dfc25_track2_val_labels' / 'dfc25_track2_val_labels' / 'target'
            xbd = source / 'xBD' / 'hold'
            ramp = np.arange(64, dtype=np.uint8).reshape(size)
            rgb = np.stack([ramp] * 3, axis=-1)

            def image(path, array):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(path)

            for index, label in enumerate((1, 0, 2)):
                name = f'bright-event_{index:08d}'
                image(val / 'pre-event' / f'{name}_pre_disaster.tif', rgb)
                image(val / 'post-event' / f'{name}_post_disaster.tif', ramp)
                image(bright_labels / f'{name}_building_damage.tif', (ramp == 7).astype(np.uint8) * label)
            for time in ('pre', 'post'):
                image(xbd / 'images' / f'xbd-event_00000000_{time}_disaster.png', rgb)
            image(xbd / 'targets' / 'xbd-event_00000000_post_disaster_target.png', ramp % 5)
            data, work = root / 'data', root / 'sessions'
            summary = prep.prepare(source, data)
            self.assertEqual(summary['bright']['negative'], 2)
            audit_path = data / 'sample_audit.json'
            audit = prep.load(audit_path)
            self.assertGreater(audit['cases'][-1]['severe_pixels'], 0)
            self.assertLess(audit['cases'][-1]['severe_pixels'], audit['cases'][-1]['damage_pixels'])

            # The Planner can call SAM on SAR, which is recorded even though no instances were returned.
            routed = []

            def segment(time, text, tile, threshold):
                routed.append(time)
                return [], {}

            def fake_planner(**kwargs):
                calls = iter([('segment_text', {'time': 't2', 'text': 'building'}),
                              ('submit', {'mask': 'empty', 'summary': 'test'}),
                              ('submit', {'mask': 'empty', 'summary': 'test'})])

                def client(payload):
                    tool, args = next(calls)
                    return {'choices': [{'message': {'role': 'assistant', 'tool_calls': [
                        {'id': 'fake-call', 'type': 'function',
                         'function': {'name': tool, 'arguments': json.dumps(args)}}
                    ]}}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}

                return Planner(client=client, **kwargs)

            not_available = data / 'hidden_audit.json'
            audit_path.rename(not_available)
            with patch.object(run, 'Planner', side_effect=fake_planner), patch.object(run, 'SegmentBackend', return_value=segment):
                run.infer(data / 'inputs', work, 'localhost', 40)
            not_available.rename(audit_path)
            self.assertEqual(routed, ['t2'] * 4)
            run.report(data, work, 40)
            metrics = prep.load(work / 'metrics.json')
            self.assertEqual(metrics['failed'], 0)
            self.assertEqual(metrics['datasets']['bright']['tool_routing']['totals']['t2_optical_seg_calls'], 3)
            self.assertEqual(metrics['datasets']['xbd']['count'], 1)
            for dataset in counts:
                self.assertTrue((work / 'viz' / dataset / 'index.html').is_file())
            public_manifest = data / 'inputs' / 'cases.json'
            for path in [public_manifest, work / 'metrics.json', *work.glob('traces/*/input.json')]:
                content = path.read_text()
                for forbidden in ('offline_gt', 'source_label', 'damage_pixels', 'severe_pixels'):
                    self.assertNotIn(forbidden, content)
            entries = prep.load(public_manifest)
            entries['bright'][0]['severe_pixels'] = 5
            prep.save(public_manifest, entries)
            with self.assertRaises(ValueError):
                prep.load_inputs(data / 'inputs')


if __name__ == '__main__':
    unittest.main()
