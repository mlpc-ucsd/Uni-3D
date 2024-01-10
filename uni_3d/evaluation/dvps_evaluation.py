import numpy as np
import os
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.utils import comm
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator

from .metrics import eval_dvpq
from .metrics.dvpq import CityscapesLoader, Front3DLoader, MatterportLoader

def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors

def vis_seg(seg_map):
    colors = make_colors()
    h,w = seg_map.shape
    color_mask = np.zeros([h,w,3])
    for seg_id in np.unique(seg_map):
        color_mask[seg_map==seg_id] = colors[seg_id%len(colors)]
    return color_mask.astype(np.uint8)

class CityscapesDPSEvaluator(CityscapesEvaluator):
    """
    Evaluate video panoptic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    LOADER = CityscapesLoader
    THING_CLASS_INDICES=slice(11, None)
    STUFF_CLASS_INDICES=slice(None, 11)

    def __init__(
        self, dataset_name, output_dir):
        super().__init__(dataset_name)
        self.output_folder = output_dir
        self.evaluator = eval_dvpq
        self.eval_frames = [1, 2, 3, 4]
        self.depth_thres = [-1, 0.5, 0.25, 0.1]

    def process(self, inputs, outputs):
        save_dir = self._temp_dir
        for input, output in zip(inputs, outputs):
            basename = os.path.basename(input['file_name'])
            pred_depth = output['depth'].to(self._cpu_device).numpy()

            panoptic_img, segments_info = output["panoptic_seg"]
            semantic_img = torch.ones_like(panoptic_img) * self.LOADER.NUM_CLASSES

            for segm in segments_info:
                semantic_img[panoptic_img == segm["id"]] = segm["category_id"]

            pan_result = torch.dstack([semantic_img, panoptic_img, torch.zeros_like(panoptic_img)])
            pred_depth = (pred_depth*256).astype(np.int32)
            pan_result = pan_result.to(self._cpu_device).numpy().astype(np.uint8)

            Image.fromarray(pred_depth).save(
                os.path.join(save_dir, basename.replace("_leftImg8bit.png", "_depth.png")))
            Image.fromarray(pan_result).save(
                os.path.join(save_dir, basename.replace("_leftImg8bit.png", "_panoptic.png")))
            Image.fromarray(vis_seg(pan_result[:,:,1])).save(
                os.path.join(save_dir, basename.replace("_leftImg8bit.png", "_panoptic_vis.png")))

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        return self.evaluate_dpq()

    def evaluate_dpq(self):
        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))
        dpq = {}
        pred_dir = self._temp_dir
        for depth_thres in self.depth_thres:
            results = self.evaluator(1, self.LOADER.NUM_CLASSES, self.LOADER(pred_dir, self._metadata.gt_dir, depth_thres))
            pq = results[0][:self.LOADER.NUM_CLASSES]
            pq_th, pq_st = pq[self.THING_CLASS_INDICES], pq[self.STUFF_CLASS_INDICES]
            pq_th_mean, pq_st_mean = pq_th.mean(), pq_st.mean()
            pq_mean = np.concatenate([pq_th, pq_st]).mean()
            print('k={}, lambda={}, result:\n'.format(1, depth_thres),
                'PQ     PQ_th  PQ_st\n',
                '{:.2f}  {:.2f}  {:.2f}'.format(
                pq_mean,
                pq_th_mean,
                pq_st_mean))
            dpq[str(depth_thres)] = {'dpq':    pq_mean,
                                     'dpq_th': pq_th_mean,
                                     'dpq_st': pq_st_mean}
        ret = OrderedDict()
        ret.update(dpq)
        ret['averages'] = {
            'dpq':    np.array([dpq[str(depth_thres)]['dpq']    for depth_thres in self.depth_thres if depth_thres > 0]).mean(),
            'dpq_th': np.array([dpq[str(depth_thres)]['dpq_th'] for depth_thres in self.depth_thres if depth_thres > 0]).mean(),
            'dpq_st': np.array([dpq[str(depth_thres)]['dpq_st'] for depth_thres in self.depth_thres if depth_thres > 0]).mean()
            }

        self._working_dir.cleanup()
        return ret


class Front3DDPSEvaluator(CityscapesDPSEvaluator):
    LOADER = Front3DLoader
    THING_CLASS_INDICES=slice(1, 10)
    STUFF_CLASS_INDICES=slice(10, None)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            save_dir = os.path.join(self._temp_dir, input['scene_id'])
            os.makedirs(save_dir, exist_ok=True)
            basename = os.path.join(save_dir, input['raw_image_id'])

            pred_depth = output['depth'].to(self._cpu_device).numpy()

            panoptic_img, segments_info = output["panoptic_seg"]
            semantic_img = torch.ones_like(panoptic_img) * self.LOADER.NUM_CLASSES

            for segm in segments_info:
                semantic_img[panoptic_img == segm["id"]] = segm["category_id"]

            pan_result = torch.dstack([semantic_img, panoptic_img, torch.zeros_like(panoptic_img)])
            Image.fromarray((pred_depth/25*256).astype(np.uint8)).save(basename + "_depth_vis.png")
            pred_depth = (pred_depth*256).astype(np.int32)
            pan_result = pan_result.to(self._cpu_device).numpy().astype(np.uint8)

            Image.fromarray(pred_depth).save(basename + "_depth.png")
            Image.fromarray(pan_result).save(basename + "_panoptic.png")
            Image.fromarray(vis_seg(pan_result[:,:,1])).save(basename + "_panoptic_vis.png")


class MatterportDPSEvaluator(Front3DDPSEvaluator):
    LOADER = MatterportLoader
