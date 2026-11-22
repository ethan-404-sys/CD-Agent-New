import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch
from torch import nn

from cd_agent.diagnostic.backbone import DummyBackbone, LoRALinear, SAM3Backbone, _load_retained
from cd_agent.diagnostic.data import MODEL_INPUT_KEYS, SecondDiagnosticDataset, collate_diagnostics
from cd_agent.diagnostic.inference import predict_episode
from cd_agent.diagnostic.losses import counterfactual_margin_loss, diagnostic_loss
from cd_agent.diagnostic.metrics import DiagnosticMetrics, hypothesis_utilization_gain
from cd_agent.diagnostic.model import HVVerifier, gated_probabilities
from cd_agent.verify import EpisodeInput


def setUpModule():
    global _threads
    _threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_threads)


class DiagnosticModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.t1 = torch.rand(2, 3, 28, 28)
        self.t2 = torch.rand_like(self.t1)
        self.mask = torch.zeros(2, 1, 28, 28, dtype=torch.bool)
        self.mask[:, :, :14] = True
        self.prompts = ['into building', 'out of building']

    def model(self, mode='hv'):
        return HVVerifier(DummyBackbone(channels=32), mode=mode)

    def test_shapes_and_directional_asymmetry_in_both_modes(self):
        for mode in ('hv', 'additive'):
            with self.subTest(mode=mode), torch.inference_mode():
                model = self.model(mode).eval()
                outputs = model(self.t1, self.t2, self.mask, self.prompts)
                reverse = model(self.t2, self.t1, self.mask, self.prompts)
                self.assertEqual(set(outputs), {'logits_fp', 'logits_fn', 'logits_has_fp', 'logits_has_fn', 'q_score'})
                self.assertEqual(outputs['logits_fp'].shape, self.mask.shape)
                self.assertEqual(outputs['logits_fn'].shape, self.mask.shape)
                for name in ('q_score', 'logits_has_fp', 'logits_has_fn'):
                    self.assertEqual(outputs[name].shape, (2,))
                self.assertTrue(((outputs['q_score'] > 0) & (outputs['q_score'] < 1)).all())
                self.assertFalse(torch.allclose(outputs['logits_fn'], reverse['logits_fn']))
                probabilities = gated_probabilities(outputs, self.mask)
                self.assertFalse(probabilities['prob_fp'][~self.mask].any())
                self.assertFalse(probabilities['prob_fn'][self.mask].any())

    def test_extreme_masks_and_backward(self):
        mask = self.mask.clone()
        mask[0] = False
        mask[1] = True
        gt = self.mask.clone()
        ignore = torch.zeros_like(mask)
        ignore[..., :2, :] = True
        for mode in ('hv', 'additive'):
            with self.subTest(mode=mode):
                model = self.model(mode)
                outputs = model(self.t1, self.t2, mask, self.prompts)
                self.assertTrue(all(t.isfinite().all() for t in outputs.values()))
                parts = diagnostic_loss(outputs, init_mask=mask, gt_clean=gt, ignore_mask=ignore,
                                        fp_mask=mask & ~gt & ~ignore, fn_mask=~mask & gt & ~ignore)
                self.assertTrue(all(t.isfinite().all() for t in parts.values()))
                parts['total'].backward()
                self.assertIsNone(model.backbone.text.weight.grad)
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(parameter.grad.isfinite().all(), name)

    def test_bce_only_zero_target_and_ignore(self):
        mask = self.mask
        logits = torch.zeros_like(mask, dtype=torch.float, requires_grad=True)
        outputs = {'logits_fp': logits, 'logits_fn': logits, 'logits_has_fp': logits[:, 0, 0, 0],
                   'logits_has_fn': logits[:, 0, 0, 0], 'q_score': logits[:, 0, 0, 0].sigmoid()}
        for ignored in (False, True):
            parts = diagnostic_loss(outputs, init_mask=mask, gt_clean=mask, ignore_mask=torch.full_like(mask, ignored),
                                    fp_mask=torch.zeros_like(mask), fn_mask=torch.zeros_like(mask))
            self.assertAlmostEqual(parts['fp'].item(), 0.0 if ignored else np.log(2), places=6)
            self.assertAlmostEqual(parts['fn'].item(), 0.0 if ignored else np.log(2), places=6)
            if ignored:
                self.assertEqual(parts['total'].item(), 0.0)

    def test_lora_and_warmup_switch(self):
        layer = nn.Linear(4, 4)
        x = torch.rand(2, 4)
        lora = LoRALinear(layer, 2, 4)
        torch.testing.assert_close(lora(x), layer(x))
        lora(x).sum().backward()
        self.assertIsNone(layer.weight.grad)
        self.assertGreater(lora.lora_b.weight.grad.abs().sum().item(), 0)
        model = self.model()
        for enabled in (False, True):
            model.set_vision_lora_trainable(enabled)
            self.assertTrue(all(p.requires_grad == enabled for p in model.backbone.vision_adaptation_parameters()))

    def test_strict_loader_rejects_shape_and_unexpected_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'small.pt'
            parts = {'kept.': nn.Linear(2, 3)}
            original = {'kept.weight': torch.randn(3, 2), 'kept.bias': torch.randn(3)}
            torch.save(dict(model={**original, 'detector.transformer.decoder.known_unused': torch.zeros(1)}), path)
            _load_retained(parts, path)
            torch.testing.assert_close(parts['kept.'].weight, original['kept.weight'])
            for entry, error in (({'kept.weight': torch.randn(3, 1)}, 'expected'),
                                 ({'kept.new_weight': torch.zeros(1)}, 'Unexpected')):
                torch.save(dict(model={**original, **entry}), path)
                with self.assertRaisesRegex(ValueError, error):
                    _load_retained(parts, path)

    def test_prompt_length_is_checked_before_vendor_forward(self):
        backbone = SAM3Backbone.__new__(SAM3Backbone)
        nn.Module.__init__(backbone)
        backbone.text = Mock(context_length=32)
        backbone.text.tokenizer.encode.side_effect = lambda prompt: prompt.split()
        with self.assertRaisesRegex(ValueError, '32 tokens'):
            backbone.encode_text([' '.join(['class'] * 31)], torch.device('cpu'))
        backbone.text.assert_not_called()

    def test_online_loads_only_the_allowlist(self):
        with tempfile.TemporaryDirectory() as folder:
            rgb = Path(folder) / 'rgb.png'
            binary = Path(folder) / 'mask.png'
            Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8)).save(rgb)
            Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(binary)
            record = dict(id='episode', image_t1=str(rgb), image_t2=str(rgb),
                          init_mask=str(binary), instruction='into building')
            for extra in ({'gt_mask': 'label.png'}, {'meta': {'gt_clean': 'label.png'}}):
                with self.assertRaisesRegex(ValueError, 'forbidden'):
                    EpisodeInput.from_mapping({**record, **extra})
            model = self.model().eval()
            with self.assertRaises(TypeError):
                predict_episode(model, {**record, 'gt_mask': 'label.png'})
            # Even direct construction cannot make ignored fields reach the model or file loaders.
            episode = EpisodeInput(**record, ref_mask='should-not-open.png', meta={'gt_mask': 'should-not-open.png'})
            out = predict_episode(model, episode)
            self.assertEqual(out['prob_fn'].shape, (1, 1, 28, 28))

    def test_dataset_manifest_matches_nested_second_labels_and_keeps_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, array in (
                ('first', np.zeros((4, 4, 3), np.uint8)), ('second', np.full((4, 4, 3), 255, np.uint8)),
                ('init', np.eye(4, dtype=np.uint8)), ('gt', np.fliplr(np.eye(4, dtype=np.uint8))),
                ('ignore', np.zeros((4, 4), np.uint8)), ('fp', np.eye(4, dtype=np.uint8)),
                ('fn', np.fliplr(np.eye(4, dtype=np.uint8))),
            ):
                Image.fromarray(array).save(root / f'{name}.png')
            manifest = root / 'manifest.jsonl'
            row = dict(id='a', dataset='SECOND', split='train', source_image_id='one', generator='private',
                       task_type='into', prompt_en='into water', image_t1='first.png', image_t2='second.png',
                       init_mask='init.png', gt_mask='gt.png', counterfactual_group_id='base',
                       labels=dict(ignore_mask='ignore.png', fp='fp.png', fn='fn.png'))
            row2 = dict(row, id='b', task_type='outof', prompt_en='out of water')
            manifest.write_text(''.join(json.dumps(entry) + '\n' for entry in (row, row2)))
            dataset = SecondDiagnosticDataset(manifest, split='train', spatial_augment=True, contour_randomize=True)
            batch = collate_diagnostics([dataset[0], dataset[1]])
            x, y = batch['inputs'], batch['targets']
            self.assertEqual(set(x), set(MODEL_INPUT_KEYS))
            self.assertFalse(x['image_t1'].any())
            self.assertTrue(x['image_t2'].all())
            self.assertTrue(torch.equal(x['init_mask'][0], x['init_mask'][1]))
            torch.testing.assert_close((x['init_mask'] & ~y['fp_mask']) | y['fn_mask'], y['gt_clean'])
            self.assertEqual(batch['cf_pairs'], [(0, 1)])
            manifest.write_text(''.join(json.dumps(entry) + '\n' for entry in (row, dict(row2, split='val'))))
            with self.assertRaisesRegex(ValueError, 'split leakage'):
                SecondDiagnosticDataset(manifest, split='train')

    def test_counterfactual_and_metrics(self):
        mask = torch.tensor([True, False, False]).reshape(1, 1, 3)
        gt_a = torch.tensor([False, True, False]).reshape_as(mask)
        gt_b = torch.tensor([True, False, False]).reshape_as(mask)
        false = torch.zeros_like(mask)
        fp, fn = mask & ~gt_a, ~mask & gt_a
        meter = DiagnosticMetrics()
        meter.update(fp, fn, mask, gt_a, false, task='into', cf_group='pair', instruction='into', zeroed_fn=false)
        meter.update(false, false, mask, gt_b, false, task='outof', cf_group='pair', instruction='outof')
        metrics = meter.summary()
        self.assertEqual(metrics['overall']['fp_f1'], 1)
        self.assertEqual(metrics['overall']['fn_iou'], 1)
        self.assertEqual(metrics['overall']['refined_iou'], 1)
        self.assertEqual(metrics['overall']['delta_iou'], 0.5)
        self.assertEqual(metrics['overall']['anchor_fpr'], 0)
        self.assertEqual(metrics['overall']['cf_flip_rate'], 1)
        self.assertEqual(metrics['by_task']['into']['hypothesis_utilization_gain'], 1)
        self.assertEqual(metrics['by_init_iou']['[0.7,1.0]']['count'], 1)
        self.assertIsNone(hypothesis_utilization_gain(false, false, gt_b, mask, false)['gain'])
        targets = {'fp': torch.stack((fp, false)), 'fn': torch.stack((fn, false))}
        correct = {f'logits_{k}': 2 * v.float() for k, v in targets.items()}
        masks = torch.stack((mask, mask))
        valid = torch.ones_like(masks)
        self.assertEqual(counterfactual_margin_loss(correct, masks, valid, targets, [(0, 1)]).item(), 0)
        incorrect = {k: -v for k, v in correct.items()}
        self.assertEqual(counterfactual_margin_loss(incorrect, masks, valid, targets, [(0, 1)]).item(), 3)


if __name__ == '__main__':
    unittest.main()
