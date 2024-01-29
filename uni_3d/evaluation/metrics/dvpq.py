import numpy as np
from PIL import Image
import six
import os
import multiprocessing as mp
from functools import partial
import json
import pyexr

MAX_INS = 1000


def _vpq_eval(element, max_ins=MAX_INS, ign_id=32, offset=256*256, num_cat=20):
    preds, gts = element

    assert isinstance(preds, list)
    assert isinstance(gts, list)
    assert len(preds) == len(gts)

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    pred_ids = np.concatenate(preds, axis=1)
    gt_ids = np.concatenate(gts, axis=1)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas)
        if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.uint32) * offset + pred_ids.astype(np.uint32)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        pred_id = int_id % offset
        pred_cat = pred_id // max_ins
        if pred_cat == ign_id:
            continue
        gt_id = int_id // offset
        gt_cat = gt_id // max_ins
        if gt_cat != pred_cat:
            continue
        union = (
            gt_areas[gt_id] + pred_areas[pred_id] - int_area -
            prediction_void_overlap(pred_id)
        )
        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = gt_id // max_ins
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = pred_id // max_ins
        if cat == ign_id:
            continue
        fp_per_class[cat] += 1

    return (iou_per_class, tp_per_class, fn_per_class, fp_per_class)


def _eval(element, loader, eval_frames, void_class_id):
    preds, gts, depth_preds, depth_gts = loader.load(*element)

    abs_rel = 0
    depth_thres = loader.depth_thres
    num_frames = loader.NUM_FRAMES
    if depth_thres > 0:
        depth_pred_cat = np.concatenate(depth_preds, axis=1)
        depth_gt_cat = np.concatenate(depth_gts, axis=1)
        depth_mask = depth_gt_cat > 0
        abs_rel = np.mean(
            np.abs(
                depth_pred_cat[depth_mask] -
                depth_gt_cat[depth_mask]) /
            depth_gt_cat[depth_mask])
        for depth_pred, depth_gt, pred in zip(depth_preds, depth_gts, preds):
            depth_mask = depth_gt > 0
            pred_in_depth_mask = pred[depth_mask]
            ignored_pred_mask = (
                np.abs(
                    depth_pred[depth_mask] -
                    depth_gt[depth_mask]) /
                depth_gt[depth_mask]) > depth_thres
            pred_in_depth_mask[ignored_pred_mask] = void_class_id * MAX_INS
            pred[depth_mask] = pred_in_depth_mask

    def _gt_process(img, max_ins=1000):
        cat = img // 1000
        ins = img % 1000
        ids = cat * max_ins + ins
        return ids.astype(np.uint32)

    gts = [_gt_process(gt, MAX_INS) for gt in gts]
    results = []
    for i in range(num_frames - eval_frames + 1):
        pred_t = preds[i: i + eval_frames]
        gt_t = gts[i: i + eval_frames]
        result = _vpq_eval([pred_t, gt_t], ign_id=void_class_id)
        results.append(result)

    iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)
    tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)
    fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)
    fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)

    return (iou_per_class, tp_per_class, fn_per_class, fp_per_class, abs_rel)


class _EvalLoader:
    NUM_FRAMES = 0

    def __init__(self, pred_dir, metadata, depth_thres):
        self.pred_dir = pred_dir
        self.metadata = metadata
        self.depth_thres = depth_thres

    def prepare(self):
        raise NotImplementedError
    
    def load(self, preds, gts, depth_preds, depth_gts):
        raise NotImplementedError
    

class Front3DLoader(_EvalLoader):
    NUM_FRAMES = 1
    NUM_CLASSES = 13
    IS_MATTERPORT = False

    def prepare(self):
        with open(self.metadata.gt_dir) as fp:
            gt_json = json.load(fp)

        preds = {}
        gts = {}
        depth_preds = {}
        depth_gts = {}

        for item in gt_json:
            scene_id, image_id = item["scene_id"], item["image_id"]
            entry = (scene_id, image_id)
            key = scene_id + '_' + image_id
            preds[key] = [entry]
            gts[key] = [entry]

            if self.depth_thres > 0:
                depth_preds[key] = [entry]
                depth_gts[key] = [entry]

        return preds, gts, depth_preds, depth_gts

    def load(self, _preds, _gts, _depth_preds, _depth_gts):
        gt_dir = self.metadata.image_root
        preds = []
        gts = []
        depth_preds = []
        depth_gts = []
        
        for scene_id, image_id in _preds:
            pred = np.array(Image.open(os.path.join(self.pred_dir, scene_id, image_id + '_panoptic.png')))
            cat, ind = pred[..., 0], pred[..., 1]
            preds.append(cat.astype(np.uint32) * MAX_INS + ind.astype(np.uint32))
            
            if self.IS_MATTERPORT:
                name, angle, rot = image_id.split("_")
                gt_segm_file = os.path.join(gt_dir, "data", scene_id, f"{name}_segmap{angle}_{rot}.mapped.npz")
            else:
                gt_segm_file = os.path.join(gt_dir, scene_id, f"segmap_{image_id}.mapped.npz")
            
            gt = np.load(gt_segm_file)["data"]
            cat_gt, ind_gt = gt[..., 0], gt[..., 1]
            if self.IS_MATTERPORT:
                ind_gt_rearranged = np.zeros_like(ind_gt)
                for dst_i, orig_i in enumerate(np.unique(ind_gt)):
                    ind_gt_rearranged[ind_gt == orig_i] = dst_i + 1
                ind_gt = ind_gt_rearranged
            cat_gt[cat_gt == 0] = self.NUM_CLASSES # mark invalid 0 class as void
            gts.append(cat_gt.astype(np.uint32) * MAX_INS + ind_gt.astype(np.uint32))

            if self.depth_thres > 0:
                depth_preds.append(np.array(Image.open(os.path.join(self.pred_dir, scene_id, image_id + '_depth.png'))))
                if self.IS_MATTERPORT:
                    depth_gt = np.array(Image.open(os.path.join(gt_dir, "depth_gen", scene_id, f"{name}_d{angle}_{rot}.png"))) / 4000
                    # HACK: mask out-of-range values as in training
                    depth_gt[depth_gt < 0.4] = 0
                    depth_gt[depth_gt > 6.0] = 0
                    depth_gt = depth_gt * 256
                else:
                    depth_gt = pyexr.read(os.path.join(gt_dir, scene_id, f"depth_{image_id}.exr")).squeeze().copy() * 256
                depth_gts.append(depth_gt.astype(np.int32))
        
        return preds, gts, depth_preds, depth_gts
    

class MatterportLoader(Front3DLoader):
    IS_MATTERPORT = True


def eval_dvpq(eval_frames, void_class_id, loader: _EvalLoader):
    preds, gts, depth_preds, depth_gts = loader.prepare()

    # Evaluation
    all_samples = []
    for seq_id in gts:
        all_samples.append([preds[seq_id], gts[seq_id], depth_preds.get(seq_id, None), depth_gts.get(seq_id, None)])
    
    N = mp.cpu_count() // 2 + 1

    _mapped_eval = partial(_eval,
                           loader=loader,
                           eval_frames=eval_frames,
                           void_class_id=void_class_id)

    with mp.Pool(processes=N) as p:
        results = p.map(_mapped_eval, all_samples)

    iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)
    tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)
    fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)
    fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)
    abs_rel = np.stack([result[4] for result in results]).mean(axis=0)
    epsilon = 1

    sq = iou_per_class / np.maximum(tp_per_class, epsilon)
    rq = tp_per_class / \
        np.maximum(tp_per_class + 0.5 * fn_per_class +
                   0.5 * fp_per_class, epsilon)
    pq = sq * rq
    return pq * 100, sq * 100, rq * 100
