import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from cd_agent.backends import SegmentBackend
from cd_agent.loop_json import run_session
from cd_agent.planner import Planner
from cd_agent.recorder import Recorder
from cd_agent.segment_cache import read_instances, read_segment_result
from cd_agent.t32_dev32 import render_reference_panel
from cd_agent.tools import Toolbox, session_schemas
from cd_agent.workers.sam3_worker import Segmenter
from .helpers import call, mask, response, rgb, ScriptClient


class _FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return _FakeTensor(self.data[key])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class T3P0FixesTests(unittest.TestCase):
    def test_render_reference_panel_preserves_holes_and_binary_panel(self):
        ref = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
        # Large foreground covering upper half with a 16x16 excluded hole inside it.
        m = np.zeros((64, 64), dtype=np.uint8)
        m[:48, :] = 255
        m[16:32, 24:40] = 0
        panel = render_reference_panel(ref, Image.fromarray(m), tile_size=64, header_h=20)
        arr = np.asarray(panel)
        self.assertEqual(panel.size, (64 * 3 + 12, 64 + 20))
        # Middle panel (x=70..134, y=20..84): hole center must remain unshaded gray (120,120,120), NOT green/cyan!
        mid = arr[20:84, 70:134]
        hole_px = mid[24, 32]
        fg_px = mid[6, 6]
        self.assertTrue(np.allclose(hole_px, [120, 120, 120], atol=2), f'Hole should be unshaded, got {hole_px}')
        self.assertGreater(int(fg_px[1]), int(fg_px[0]) + 40, f'Foreground should be green-tinted, got {fg_px}')
        # Right panel (x=140..204, y=20..84): binary white=foreground, black=hole/background.
        right = arr[20:84, 140:204]
        self.assertEqual(int(right[6, 6, 0]), 255)
        self.assertEqual(int(right[24, 32, 0]), 0)
        self.assertEqual(int(right[56, 32, 0]), 0)

    def test_sam3_worker_saves_semantic_and_presence_even_when_instances_empty(self):
        class Processor:
            def set_image(self, image):
                self.shape = (image.height, image.width)
                return {}

            def set_text_prompt(self, prompt, state):
                h, w = self.shape
                sem = np.zeros((1, 1, h, w), dtype=np.float32)
                sem[0, 0, 10:30, 10:30] = 0.90
                return dict(
                    scores=_FakeTensor([]),
                    masks=_FakeTensor(np.zeros((0, 1, h, w), dtype=bool)),
                    semantic_mask_logits=_FakeTensor(sem),
                    presence_score=_FakeTensor(0.80),
                )

        class Engine:
            autocast = staticmethod(nullcontext)

        class Torch:
            inference_mode = staticmethod(nullcontext)

        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules', torch=Torch()), patch(
            'cd_agent.workers.sam3_worker.load_rgb', return_value=rgb()
        ):
            worker = Segmenter(Processor(), Engine(), d)
            res = worker.infer('dev', 'case-01.png', 't1', 'bare soil', 0, .1)
            self.assertTrue(res['ok'])
            # Instance mode still empty, but extra metadata reports presence_score and semantic_area.
            inst_rows, inst_extra = read_segment_result(res['path'], 0.3, mode='instance')
            self.assertEqual(len(inst_rows), 0)
            self.assertAlmostEqual(inst_extra['presence_score'], 0.80, places=2)
            self.assertEqual(inst_extra['semantic_area'], 400)
            self.assertEqual(read_instances(res['path'], 0.3), [])
            # Semantic mode returns the 20x20 region (0.9 * 0.8 = 0.72 >= 0.3).
            sem_rows, sem_extra = read_segment_result(res['path'], 0.3, mode='semantic')
            self.assertEqual(len(sem_rows), 1)
            self.assertEqual(sem_rows[0]['area'], 400)
            self.assertAlmostEqual(sem_rows[0]['score'], 0.72, delta=0.02)
            # High threshold above 0.72 filters it out cleanly (presence gating works).
            hi_rows, _ = read_segment_result(res['path'], 0.85, mode='semantic')
            self.assertEqual(len(hi_rows), 0)
            # Fused mode matches semantic when instances are empty.
            fused_rows, _ = read_segment_result(res['path'], 0.3, mode='fused')
            self.assertEqual(len(fused_rows), 1)
            self.assertEqual(fused_rows[0]['area'], 400)

    def test_explicit_answer_keeps_drafts_separate_and_sam3_semantic_schema(self):
        calls = []

        def fake_segment(time, text, tile, score_thr, mode='instance'):
            calls.append((time, text, mode))
            m = mask((10, 10, 26, 26)) if time == 't1' else mask((20, 20, 36, 36))
            idx = np.flatnonzero(m).astype(np.int32)
            rows = [dict(id=1, idx=idx, area=int(idx.size), score=0.8, bbox=[10, 10, 26, 26])]
            return rows, dict(backend='fake', mode=mode, presence_score=0.85, semantic_area=int(idx.size))

        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, segment=fake_segment, init_mask=mask((2, 2, 8, 8)),
                          explicit_answer=True, segment_pair=True, sam3_semantic=True)
            schemas = session_schemas(box.session, answer_state=True, explicit_answer=True,
                                      segment_pair=True, sam3_semantic=True)
            by_name = {s['function']['name']: s['function'] for s in schemas}
            self.assertIn('mode', by_name['segment_text']['parameters']['properties'])
            self.assertIn('mode', by_name['segment_text_pair']['parameters']['properties'])
            self.assertFalse(by_name['to_mask']['parameters']['properties']['set_current']['default'])

            # segment_text_pair with mode='semantic' surfaces presence_score & semantic_area.
            r_pair = box.execute('segment_text_pair', {'text': 'bare soil', 'score_thr': 0.3, 'mode': 'semantic'})
            self.assertTrue(r_pair.public['ok'])
            self.assertEqual(calls, [('t1', 'bare soil', 'semantic'), ('t2', 'bare soil', 'semantic')])
            self.assertAlmostEqual(r_pair.public['results']['t1']['presence_score'], 0.85, places=3)
            self.assertEqual(r_pair.public['results']['t1']['semantic_area'], 256)

            # to_mask without set_current creates a draft and does NOT overwrite current_mask ('mask#0').
            inst_t1 = r_pair.public['results']['t1']['inst']
            r_draft = box.execute('to_mask', {'sets': [inst_t1]})
            self.assertTrue(r_draft.public['ok'])
            draft_key = r_draft.public['mask']
            self.assertEqual(box.session.current_mask, 'mask#0')
            # state_summary includes area for masks and n/area for instances.
            summary = box.state_summary()
            inst_entry = next(h for h in summary['handles'] if h['handle'] == inst_t1)
            self.assertEqual(inst_entry['n'], 1)
            self.assertEqual(inst_entry['area'], 256)
            # Explicit promotion via set_answer updates current_mask.
            self.assertTrue(box.execute('set_answer', {'mask': draft_key}).public['ok'])
            self.assertEqual(box.session.current_mask, draft_key)

    def test_run_session_records_explicit_answer_and_sam3_semantic(self):
        def fake_segment(time, text, tile, score_thr, mode='instance'):
            m = mask((10, 10, 26, 26))
            idx = np.flatnonzero(m).astype(np.int32)
            return [dict(id=1, idx=idx, area=int(idx.size), score=0.9, bbox=[10, 10, 26, 26])], dict(mode=mode, presence_score=0.9, semantic_area=256)

        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, segment=fake_segment, explicit_answer=True, segment_pair=True, sam3_semantic=True)
            script = [
                response([call('segment_text_pair', {'text': 'dirt', 'mode': 'semantic'})]),
                response([call('to_mask', {'sets': ['inst#0']})]),
                response([call('set_answer', {'mask': 'mask#2'})]),
                response([call('submit', {'mask': 'mask#2', 'summary': 'Done.'})]),
            ]
            recorder = Recorder(Path(d), 'dev', 'case-01', gt=mask((10, 10, 26, 26)))
            final = run_session(box, Planner(client=ScriptClient(script), sleep=lambda _: None), recorder,
                                task='T3.2', prompt_version='refcd', answer_state=True,
                                explicit_answer=True, segment_pair=True, sam3_semantic=True)
            self.assertEqual(final['submitted_by'], 'model')
            self.assertEqual(recorder.steps[1]['calls'][0]['set_current'], False)


if __name__ == '__main__':
    unittest.main()
