"""Unit tests for CD-Agent verification primitives and visual evidence pack."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cd_agent.evidence import build_evidence_page, select_evidence_cases
from cd_agent.verify import (
    EpisodeInput,
    check_binary_mask,
    check_diagnostic_labels,
    check_split_isolation,
    verify_run_dir,
    write_run_contract,
)


class VerifyBaseTest(unittest.TestCase):
    def test_episode_input_blocks_gt_leakage(self):
        ok = EpisodeInput.from_mapping({
            'id': 'case_01',
            'image_t1': 'a.png',
            'image_t2': 'b.png',
            'instruction': 'Find new buildings.',
        })
        self.assertEqual(ok.id, 'case_01')

        with self.assertRaises(ValueError):
            EpisodeInput.from_mapping({
                'id': 'case_02',
                'image_t1': 'a.png',
                'image_t2': 'b.png',
                'instruction': 'Find new buildings.',
                'gt_mask': 'gt.png',
            })

        with self.assertRaises(ValueError):
            EpisodeInput.from_mapping({
                'id': 'case_03',
                'image_t1': 'a.png',
                'image_t2': 'b.png',
                'instruction': 'Find new buildings.',
                'meta': {'raw_task_gt': 'secret.png'},
            })

    def test_split_isolation(self):
        res = check_split_isolation({'train': ['a', 'b'], 'val': ['c'], 'test': ['d']})
        self.assertTrue(res['ok'])
        with self.assertRaises(ValueError):
            check_split_isolation({'train': ['a', 'b'], 'val': ['b', 'c']})

    def test_binary_mask_and_diagnostic_axioms(self):
        m = np.zeros((16, 16), dtype=bool)
        g = np.zeros((16, 16), dtype=bool)
        m[2:8, 2:8] = True
        g[4:10, 4:10] = True
        ign = np.zeros((16, 16), dtype=bool)
        ign[0:2, 0:2] = True
        fp = m & ~g & ~ign
        fn = ~m & g & ~ign

        stats = check_diagnostic_labels(m, g, fp, fn, ign)
        self.assertEqual(stats['fp_area'], int(fp.sum()))
        self.assertEqual(stats['fn_area'], int(fn.sum()))

        # Violating FP subset of M must fail
        bad_fp = fp.copy()
        bad_fp[12, 12] = True
        with self.assertRaises(ValueError):
            check_diagnostic_labels(m, g, bad_fp, fn, ign)

        # Non-finite float mask must fail
        nan_mask = np.zeros((8, 8), dtype=np.float32)
        nan_mask[0, 0] = np.nan
        with self.assertRaises(ValueError):
            check_binary_mask(nan_mask)

    def test_run_contract_write_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / '20261122_smoke'
            write_run_contract(
                run_dir,
                track='verifier_diag',
                run_name='20261122_smoke',
                config={'lr': 1e-4},
                metrics={'count': 10, 'failed': 0, 'f1': 0.82},
            )
            rep = verify_run_dir(run_dir, expected_count=10)
            self.assertTrue(rep['ok'])

            with self.assertRaises(ValueError):
                verify_run_dir(run_dir, expected_count=12)

            with self.assertRaises(ValueError):
                write_run_contract(
                    run_dir,
                    track='verifier_diag',
                    run_name='bad',
                    config={},
                    metrics={'count': 1, 'f1': float('nan')},
                )

    def test_evidence_pack_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'evidence'
            cases = []
            for i in range(6):
                t1 = np.full((32, 32, 3), 80 + i * 10, dtype=np.uint8)
                t2 = np.full((32, 32, 3), 120 + i * 10, dtype=np.uint8)
                gt = np.zeros((32, 32), dtype=bool)
                pred = np.zeros((32, 32), dtype=bool)
                gt[4:16, 4:16] = True
                if i == 0:
                    pred[4:16, 4:16] = True
                elif i == 1:
                    pred[4:12, 4:16] = True
                elif i == 2:
                    pred[0:24, 0:24] = True
                else:
                    pred[10:16, 10:16] = True
                cases.append({
                    'id': f'case_{i:02d}',
                    'image_t1': t1,
                    'image_t2': t2,
                    'pred_mask': pred,
                    'gt_mask': gt,
                    'summary': f'step count = {i + 1}',
                })
            page = build_evidence_page(cases, out, title='Smoke Review', metrics={'f1': 0.75})
            self.assertTrue(page.is_file())
            sel = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
            self.assertGreaterEqual(len(sel), 4)
            self.assertEqual(len(sel), len({r['id'] for r in sel}))
            self.assertTrue(any((out / 'assets').glob('*_err_t1.png')))
            self.assertTrue(any((out / 'assets').glob('*_err_t2.png')))
            html = page.read_text(encoding='utf-8')
            self.assertIn('Err@T1', html)
            self.assertIn('Err@T2', html)
            picked = select_evidence_cases([{'id': 'a', 'f1': 0.9, 'fp': 1, 'fn': 1}], per_bucket=2)
            self.assertEqual(len(picked), 1)


if __name__ == '__main__':
    unittest.main()
