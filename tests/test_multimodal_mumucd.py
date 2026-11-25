"""Local tests; no NetCDF decoding, credentials, network, or GPU required."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from experiments.multimodal_probe import mumucd_prepare as prep
from experiments.multimodal_probe import mumucd_run as run


class TiranaProbeTest(unittest.TestCase):
    def test_grid_covers_scene_once(self):
        cover = np.zeros((prep.SCENE, prep.SCENE), np.uint8)
        grid = list(prep.tiles())
        self.assertEqual(len(grid), 9)
        self.assertEqual(grid[0], ('tirana-r0c0', (0, 0, 512, 512)))
        self.assertEqual(grid[-1], ('tirana-r2c2', (1024, 1024, 1536, 1536)))
        for _, (x0, y0, x1, y1) in grid:
            self.assertEqual((x1 - x0, y1 - y0), (512, 512))
            cover[y0:y1, x0:x1] += 1
        self.assertTrue((cover == 1).all())

    def test_stretch_ignores_nodata_and_clips_per_band(self):
        base = np.arange(101, dtype=np.float32).reshape(1, 101)
        rgb = np.stack((base, base * 2 + 7, base - 100), axis=-1)
        rgb[0, 0, 0] = np.nan
        result, meta = prep.stretch_rgb(rgb)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(meta[0]['invalid_pixels'], 1)
        self.assertEqual(meta[1]['low'], 11)
        self.assertEqual(meta[2]['high'], -2)
        self.assertEqual(result[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(result[0, -1].tolist(), [255, 255, 255])
        self.assertEqual(result[0, 50, 1:].tolist(), [128, 128])

    def test_gt_values_and_nodata(self):
        raw = np.array([[0, 1, -2147483647]], dtype=np.int32)
        gt, valid = prep.decode_gt(raw, -2147483647)
        self.assertEqual(gt.tolist(), [[False, True, False]])
        self.assertEqual(valid.tolist(), [[True, True, False]])
        self.assertEqual(prep.value_counts(raw), {'0': 1, '1': 1, '-2147483647': 1})
        with self.assertRaisesRegex(ValueError, 'Unexpected GT values'):
            prep.decode_gt(np.array([[0, 255]]), None)

    def test_counting_records_sar_tool_routes_and_python_mentions(self):
        calls = [
            dict(tool='segment_text', args={'time': 't1'}, ok=True),
            dict(tool='segment_text', args={'time': 't2'}, ok=False),
            dict(tool='segment_prompt', args={'time': 't2'}, ok=True),
            dict(tool='python', args={'code': 'print(images["t2"].mean(), images["t1"].mean())'}, ok=True),
            dict(tool='python', args={'code': '# images["t2"]\nprint("images[\\\"t1\\\"]")'}, ok=True),
            dict(tool='view', args={}, ok=False, status='skipped_due_to_submit'),
        ]
        stats = run.tool_routing([{'calls': calls}])
        self.assertEqual(stats['segmentation_by_requested_time']['segment_text'], {'t1': 1, 't2': 1})
        self.assertEqual(stats['outcomes']['segment_text'], {'ok': 1, 'error': 1})
        self.assertEqual(stats['outcomes']['view'], {'skipped_due_to_submit': 1})
        self.assertEqual(stats['python_literal_image_mentions'], {'t1+t2': 1, 'unspecified': 1})
        self.assertEqual(run.python_route('print(images['), 'unspecified')

    def test_sensor_prompt_and_empty_mask_conventions(self):
        for phrase in ('10 m Sentinel-2', '10 m Sentinel-1', 'VV - VH', 'B4 (red)', 'images["t2"]'):
            self.assertIn(phrase, prep.INSTRUCTION)
        self.assertEqual(run.BUDGET, 14)
        self.assertEqual(run.MAX_COST, 20)
        self.assertEqual(run.pixel_metrics(0, 0, 0)['f1'], 1)
        self.assertEqual(run.pixel_metrics(1, 1, 2)['recall'], 1 / 3)

    def test_run_session_needs_no_offline_labels_and_keeps_t2_tools(self):
        from cd_agent.planner import Planner

        def fake_planner(_, payload):
            self.assertEqual(payload['model'], run.MODEL)
            schemas = {tool['function']['name']: tool['function'] for tool in payload['tools']}
            self.assertIn('python', schemas)
            self.assertEqual(schemas['segment_text']['parameters']['properties']['time']['enum'], ['t1', 't2'])
            user_text = payload['messages'][1]['content'][0]['text']
            self.assertIn('Sentinel-1 SAR', user_text)
            return {'id': 'test', 'choices': [{'message': {'role': 'assistant', 'tool_calls': [{
                'id': 'submit', 'type': 'function', 'function': {
                    'name': 'submit', 'arguments': json.dumps({'mask': 'empty', 'summary': 'No change'}),
                },
            }]}}]}

        with TemporaryDirectory() as tmp, patch.object(Planner, '_http', fake_planner):
            root = Path(tmp)
            inputs = root / 'inputs'
            episodes = [dict(id=case, image_t1=f'{case}/t1.png', image_t2=f'{case}/t2.png', instruction=prep.INSTRUCTION)
                        for case, _ in prep.tiles()]
            case = episodes[0]['id']
            (inputs / case).mkdir(parents=True)
            prep.write_json(inputs / 'cases.json', episodes)
            for name in ('t1.png', 't2.png'):
                Image.new('RGB', (512, 512)).save(inputs / case / name)
            run.infer(inputs, root / 'traces', 'unused', 1, 20)
            trace = root / 'traces' / case
            final = run.read_json(trace / 'final.json')
            self.assertEqual(final['submitted_by'], 'model')
            self.assertIsNone(final['eval'])
            self.assertNotIn('gt_mask_path', run.read_json(trace / 'input.json'))

    def test_stage_label_blind_inputs_then_score_with_ignore(self):
        with TemporaryDirectory() as tmp, patch.object(prep, 'SCENE', 6), patch.object(prep, 'TILE', 2), patch.object(run, 'TILE', 2):
            root = Path(tmp)
            rgb = np.full((6, 6, 3), 127, np.uint8)
            gt = np.zeros((6, 6), np.int32)
            gt[0, 0], gt[0, 1] = 1, -999
            dw = np.zeros_like(gt)
            with patch.object(prep, 'read_sources', return_value=(
                dict(t1=rgb, t2=rgb, gt=gt, dw_before=dw, dw_after=gt), {'gt': {'nodata': -999}},
            )):
                prep.prepare(root / 'not-used', root)

            inputs = root / 'inputs'
            records = run.read_json(inputs / 'cases.json')
            self.assertEqual(len(records), 9)
            self.assertEqual(set(records[0]), {'id', 'image_t1', 'image_t2', 'instruction'})
            self.assertEqual(len(list(inputs.rglob('*.png'))), 18)
            self.assertFalse(any('gt' in path.name or 'audit' in path.name for path in inputs.rglob('*')))
            self.assertEqual(run.read_json(root / 'sample_audit.json')['gt_positive_tiles'], 1)
            case = records[0]['id']
            trace = root / 'traces' / case
            trace.mkdir(parents=True)
            Image.fromarray(np.array([[255, 255], [255, 0]], np.uint8)).save(trace / 'prediction.png')
            prep.write_json(trace / 'final.json', dict(mask_path='prediction.png', error=None, planning_rounds=1,
                                                      summary='test', submitted_by='model', termination_reason='submit',
                                                      planner_api_attempts=1, ask_api_attempts=0, cost_cny=.001))
            (trace / 'steps.jsonl').write_text(json.dumps({'calls': []}) + '\n')
            run.report(root, 1, run.MAX_COST)
            result = run.read_json(root / 'metrics.json')
            self.assertEqual(result['count'], 1)
            self.assertEqual(result['micro'], run.pixel_metrics(1, 1, 0))
            self.assertEqual(result['macro']['f1'], result['micro']['f1'])
            self.assertTrue((root / 'viz' / 'index.html').is_file())
            records[0]['gt_positive_pixels'] = 1
            prep.write_json(inputs / 'cases.json', records)
            with self.assertRaisesRegex(ValueError, 'nonstandard inference'):
                list(run.load_inputs(inputs))


if __name__ == '__main__':
    unittest.main()
