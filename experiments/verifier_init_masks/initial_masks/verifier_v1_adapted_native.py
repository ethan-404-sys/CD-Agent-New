"""V4 task adapters on native DynamicEarth proposals and ReA model components.

Only RGB pairs and frozen task parameters enter inference. Related predictions
retain the V3 native route. Direction/transition are explicit task adaptations,
with sufficient model-only state for independent offline reconstruction.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from verifier_v1_ovcd_methods import (
    CHECKPOINT, MCI_WEIGHTS, DynamicAdapter, ReAAdapter, planner_mask,
)

FULL_VOCABULARY = [
    'background', 'low vegetation', 'ground,bareland', 'tree,forest',
    'water,river,pond,sea', 'building', 'sports field',
]


def select_labels(spec, labels1, labels2):
    """Select units before spatial union; labels must share the same unit index."""
    kind, params = spec['task_type'], spec['task_params']
    if kind == 'transition':
        return (labels1 == params['source_class']) & (labels2 == params['target_class'])
    category = params['class_id']
    if kind == 'related': return (labels1 == category) | (labels2 == category)
    if kind == 'into': return (labels2 == category) & (labels1 != category)
    if kind == 'outof': return (labels1 == category) & (labels2 != category)
    raise ValueError(kind)


def union_selected(proposals, selected, shape):
    return np.any(proposals[selected], axis=0) if selected.any() else np.zeros(shape, bool)


def finalize(adapter, row, state, basis, masks, metadata):
    path = adapter.work / f'{row["id"]}_model_state.npz'
    np.savez(path, **state)
    metadata.update(
        model_state=str(path.resolve()), task_basis=basis,
        input_source_paths=[row['image_t1'], row['image_t2']], gt_used=False,
        adaptation_version='task_adapted_v4',
    )
    return masks, metadata


def _proposal_scores(model, seg_logits, proposals):
    """Record exactly the native SegEarth class aggregation and proposal mean.

    Native postprocess_result still produces the actual labels; this recorder
    only exposes its scores and checks that their argmax yields those labels.
    """
    import torch
    scores = (seg_logits[0] * model.logit_scale).softmax(0)
    count = int(model.num_classes)
    if count != model.num_queries:
        indices = torch.nn.functional.one_hot(model.query_idx).T.view(count, model.num_queries, 1, 1)
        scores = (scores.unsqueeze(0) * indices).max(1)[0]
    means, support = [], []
    for proposal in proposals:
        mask = torch.as_tensor(proposal, device=scores.device, dtype=torch.uint8)
        mask = torch.nn.functional.interpolate(mask[None, None], size=scores.shape[1:], mode='nearest')[0, 0].bool()
        support.append(int(mask.sum().item()))
        means.append(scores[:, mask].mean(dim=1).float().cpu().numpy())
    return (np.asarray(means, np.float32).reshape(len(proposals), count),
            np.asarray(support,np.int64),np.asarray(scores.shape[1:],np.int64))


class _ProposalRecorder:
    def __init__(self, model, canonical_ids):
        self.model = model
        self.canonical_ids = np.asarray(canonical_ids, np.uint8)
        self.scores, self.labels = [], []
        self.supports, self.grids = [], []

    def predict(self, *args, **kwargs):
        model = self.model
        original = model.postprocess_result
        captured = []

        def record(seg_logits, data_samples, proposal_masks=None):
            labels = original(seg_logits, data_samples, proposal_masks)
            captured.append(_proposal_scores(model, seg_logits, proposal_masks))
            return labels

        model.postprocess_result = record
        try:
            labels = model.predict(*args, **kwargs)
        finally:
            model.postprocess_result = original
        if len(captured) != 1: raise RuntimeError('Expected one native SegEarth postprocess call')
        scores, support, grid = captured[0]
        raw_labels = np.asarray(labels, np.uint8)
        reconstructed = scores.argmax(1).astype(np.uint8)
        reconstructed[scores.max(1) < model.prob_thd] = model.bg_idx
        if not np.array_equal(raw_labels, reconstructed):
            raise RuntimeError('Recorded proposal means differ from native labels')
        canonical = np.zeros((len(scores), 7), np.float32)
        canonical[:, self.canonical_ids] = scores
        self.scores.append(canonical)
        self.labels.append(self.canonical_ids[raw_labels])
        self.supports.append(support)
        self.grids.append(grid)
        return labels


class DynamicAdaptedAdapter(DynamicAdapter):
    """Native binary proposal labels for direction; full-class proposal transition."""

    def infer(self, row):
        from segment_anything.utils.amg import rle_to_mask
        from dynamic_earth.comparator.bi_match import bitemporal_match
        from dynamic_earth.identifier.utils import identify
        images = [np.asarray(Image.open(row[f'image_t{i}']).convert('RGB')) for i in (1, 2)]
        shape = images[0].shape[:2]
        primary = next(t['task_params']['class_id'] for t in row['pilot_tasks'] if t['task_type'] == 'related')
        key = row['dataset'] if row['dataset'] != 'SECOND' else str(primary)
        cfg = dict(self.configs[key])
        if row['dataset'] == 'WHU-CD': cfg['points_per_side'] = int(np.ceil(max(shape) / 32))
        points = cfg['points_per_side']
        self.mp.make_mask_generator(
            model=self.sam, points_per_side=points, points_per_batch=64,
            pred_iou_thresh=.5, stability_score_thresh=.95,
            stability_score_offset=.9, box_nms_thresh=.7, min_mask_region_area=0,
        )
        self.mp.set_hyperparameters(match_hist=False, area_thresh=.8)
        proposal_data, first_count = self.mp.forward(*images)
        proposals = np.asarray([rle_to_mask(r).astype(bool) for r in proposal_data['rles']], bool).reshape(-1, *shape)
        total_proposals = len(proposals)
        if len(proposals):
            proposals, first_count = bitemporal_match(
                *images, proposals, self.dino, self.processor, first_count, device='cuda',
                change_confidence_threshold=cfg['change_confidence_threshold'],
                model_config=dict(model_type='DINOv2', feature_dim=768, patch_size=14),
            )
        state = dict(proposals=proposals, B=np.any(proposals, axis=0) if len(proposals) else np.zeros(shape, bool),
                     proposal_source_phase=np.r_[np.ones(first_count, np.uint8), np.full(len(proposals) - first_count, 2, np.uint8)])
        binary_classes = sorted({t['task_params']['class_id'] for t in row['pilot_tasks'] if t['task_type'] != 'transition'})
        configs = dict(change_domain=cfg, primary_class=primary, proposal_count=total_proposals, changed_proposals=len(proposals))
        for category in binary_classes:
            rcfg = dict(self.configs[row['dataset'] if row['dataset'] != 'SECOND' else str(category)])
            rcfg['points_per_side'] = points
            model, processor = self.identifier_for(rcfg)
            recorder = _ProposalRecorder(model, [0, category])
            if len(proposals):
                native_masks, _ = identify(
                    *images, proposals, first_count, recorder, processor,
                    device='cuda', is_instance_class=rcfg['is_instance_class'],
                )
                scores1, scores2 = recorder.scores
                labels1, labels2 = recorder.labels
                supports1,supports2=recorder.supports
                grids1,grids2=recorder.grids
            else:
                native_masks = proposals
                scores1 = scores2 = np.empty((0, 7), np.float32)
                labels1 = labels2 = np.empty(0, np.uint8)
                supports1=supports2=np.empty(0,np.int64)
                grids1=grids2=np.asarray([rcfg['identifier_resize']]*2,np.int64)
            gate = labels1 != labels2
            if rcfg['is_instance_class']:
                source_is_target = np.r_[labels1[:first_count] == category, labels2[first_count:] == category]
                gate &= source_is_target
            prefix = f'binary_{category}'
            state.update({f'{prefix}_scores1': scores1, f'{prefix}_scores2': scores2,
                          f'{prefix}_labels1': labels1, f'{prefix}_labels2': labels2,
                          f'{prefix}_support1':supports1,f'{prefix}_support2':supports2,
                          f'{prefix}_grid1':grids1,f'{prefix}_grid2':grids2,
                          f'{prefix}_related_gate': gate})
            expected_native = np.any(native_masks, axis=0) if len(native_masks) else np.zeros(shape, bool)
            if not np.array_equal(union_selected(proposals, gate, shape), expected_native):
                raise RuntimeError('Native related proposal selection changed')
            state[f'related_{category}'] = expected_native
            configs[f'binary_{category}'] = dict(rcfg, prob_thd=float(model.prob_thd))
            del recorder
        if any(t['task_type'] == 'transition' for t in row['pilot_tasks']):
            fullcfg = dict(cfg, vocabulary=list(FULL_VOCABULARY))
            model, processor = self.identifier_for(fullcfg)
            if len(proposals):
                recorder = _ProposalRecorder(model, list(range(7)))
                for image in images:
                    import torch
                    tensor = processor(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode(): recorder.predict(tensor, data_samples=None, proposal_masks=proposals)
                scores1, scores2 = recorder.scores
                labels1, labels2 = recorder.labels
                supports1,supports2=recorder.supports
                grids1,grids2=recorder.grids
                del recorder
            else:
                scores1 = scores2 = np.empty((0, 7), np.float32)
                labels1 = labels2 = np.empty(0, np.uint8)
                supports1=supports2=np.empty(0,np.int64)
                grids1=grids2=np.asarray([fullcfg['identifier_resize']]*2,np.int64)
            state.update(full_scores1=scores1, full_scores2=scores2, full_labels1=labels1, full_labels2=labels2,
                         full_support1=supports1,full_support2=supports2,full_grid1=grids1,full_grid2=grids2)
            configs['transition_fullclass'] = dict(fullcfg, prob_thd=float(model.prob_thd))
        basis, masks = [], []
        for task in row['pilot_tasks']:
            kind = task['task_type']
            prefix = 'full' if kind == 'transition' else f'binary_{task["task_params"]["class_id"]}'
            labels1, labels2 = state[f'{prefix}_labels1'], state[f'{prefix}_labels2']
            selected = select_labels(task, labels1, labels2)
            item = dict(
                recipe_type='proposal_labels', route='official_related' if kind == 'related' else 'task_adapted',
                semantic_mode='proposal_fullclassargmax' if kind == 'transition' else 'native_proposal_binary',
                selection_mode=kind, proposal_masks_key='proposals',
                phase1_labels_key=f'{prefix}_labels1', phase2_labels_key=f'{prefix}_labels2',
                phase1_scores_key=f'{prefix}_scores1', phase2_scores_key=f'{prefix}_scores2', conf_threshold=0.0,
                phase1_support_key=f'{prefix}_support1',phase2_support_key=f'{prefix}_support2',
                phase1_grid_key=f'{prefix}_grid1',phase2_grid_key=f'{prefix}_grid2',
            )
            if kind == 'related':
                item['proposal_gate_key'] = f'{prefix}_related_gate'
                selected &= state[item['proposal_gate_key']]
            masks.append(union_selected(proposals, selected, shape))
            basis.append(item)
        return finalize(self, row, state, basis, masks, dict(
            official_repository='https://github.com/likyoo/DynamicEarth', configs=configs,
            weights={name: str(path) for name, path in MCI_WEIGHTS.items()},
            task_adapter='Native per-proposal class-score means; choose proposal by temporal task, then union',
            related_route='Unchanged V3 native binary identify; WHU point spacing 32 pixels',
        ))


def _component_scores(dense, component_map, count):
    """Canonical raw class-score means. Index zero is reserved for background."""
    sizes = np.bincount(component_map.ravel(), minlength=count + 1)
    result = np.zeros((count + 1, 7), np.float32)
    for category in range(7):
        totals = np.bincount(component_map.ravel(), weights=dense[category].ravel(), minlength=count + 1)
        result[1:, category] = (totals[1:] / sizes[1:]).astype(np.float32)
    return result


class ReAAdaptedAdapter(ReAAdapter):
    """Keep native B; classify its connected components from native phase scores."""

    def infer(self, row):
        from scipy import ndimage
        from ovcd_pipeline import OpenVocabularyChangeDetector
        original = OpenVocabularyChangeDetector.detect_change
        captured = []

        def record(detector, result1, result2, *args, **kwargs):
            captured.append((result1, result2))
            return original(detector, result1, result2, *args, **kwargs)

        # Capture the same native call that generates related; restore immediately.
        OpenVocabularyChangeDetector.detect_change = record
        try:
            _, metadata = ReAAdapter.infer(self, dict(row, agent_semantics=False))
        finally:
            OpenVocabularyChangeDetector.detect_change = original
        if len(captured) != 1: raise RuntimeError('Expected one native ReA detector call')
        results1, results2 = captured[0]
        cd = self.agent_cd
        mapping = metadata['native_to_canonical']
        shape = tuple(cd.pred_t1.shape)
        state = dict(B=cd.change_mask.cpu().numpy().astype(bool))
        for phase, result in enumerate((results1, results2), 1):
            native = result['seg_logits_cls'].float().cpu().numpy()
            dense = np.zeros((7, *shape), np.float32)
            dense[0] = native[0]
            labels = np.zeros(shape, np.uint8)
            native_labels = result['pred_label'].cpu().numpy()
            for source, target in mapping.items():
                dense[target] = native[source]
                labels[native_labels == source] = target
            state[f'scores{phase}'] = dense
            state[f'Y{phase}'] = labels
            presence = np.zeros(7, np.float32)
            raw_presence = result['presence_scores_cls'].float().cpu().numpy()
            presence[0] = raw_presence[0]
            for source, target in mapping.items(): presence[target] = raw_presence[source]
            state[f'presence{phase}'] = presence
        for source, target in mapping.items(): state[f'related_{target}'] = np.asarray(cd.per_class_change[source], bool)
        components, count = ndimage.label(state['B'], structure=np.ones((3, 3), bool))
        state['components'] = components.astype(np.int32)
        threshold = float(metadata['config']['prob_thd'])
        for phase in (1, 2):
            scores = _component_scores(state[f'scores{phase}'], components, count)
            labels = scores.argmax(1).astype(np.uint8)
            labels[scores.max(1) < threshold] = 0
            state[f'component_scores{phase}'] = scores
            state[f'component_labels{phase}'] = labels
        basis, masks = [], []
        for task in row['pilot_tasks']:
            kind = task['task_type']
            if kind == 'related':
                category = task['task_params']['class_id']
                item = dict(recipe_type='dense_boolean', route='official_related', selection_mode=kind,
                            semantic_mode='fullclassargmax', B_key=f'related_{category}', Y1_key='Y1', Y2_key='Y2')
                mask = planner_mask(task, state[f'related_{category}'], state['Y1'], state['Y2'])
            else:
                item = dict(recipe_type='component_labels', route='task_adapted', selection_mode=kind,
                            semantic_mode='native_component_mean_fullclass', component_map_key='components', B_key='B',
                            phase1_labels_key='component_labels1', phase2_labels_key='component_labels2',
                            phase1_scores_key='component_scores1', phase2_scores_key='component_scores2',
                            dense_phase1_scores_key='scores1', dense_phase2_scores_key='scores2', conf_threshold=threshold)
                selected = select_labels(task, state['component_labels1'], state['component_labels2'])
                selected[0] = False
                mask = selected[components]
            masks.append(mask)
            basis.append(item)
        metadata.update(
            weights=str(CHECKPOINT),
            task_adapter='Native reliable B retained; full-class phase score means classify each 8-connected B component',
            component_config=dict(connectivity=8, aggregation='mean native class logits', conf_threshold=threshold,
                                  native_change_threshold_unchanged=True, native_postprocess_unchanged=True),
            related_route='Unchanged native ReA per-class refined output',
        )
        return finalize(self, row, state, basis, masks, metadata)
