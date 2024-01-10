from typing import List, Union
import copy
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["CityscapesDPSDatasetMapper"]

VOID = 32

class CityscapesDPSDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        ignore_label,
        size_divisibility,
        dataset_name: str,
        config,
        depth_bound: bool = True,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility,
        )
        self.depth_bound = depth_bound
        self.dataset_name = dataset_name
        self.config = config

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = MaskFormerSemanticDatasetMapper.from_config(cfg, is_train)
        ret["depth_bound"] = cfg.INPUT.DEPTH_BOUND
        ret["dataset_name"] = list(cfg.DATASETS.TRAIN)[0]
        ret["config"] = cfg
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
            dataset_dict keys:
                file_name
                image_id
                vps_label_file_name
                depth_label_file_name
                next_frame
        """
        assert self.is_train, "CityscapesDPSDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "vps_label_file_name" in dataset_dict:
            pan_seg_gt = np.asarray(Image.open(dataset_dict.pop("vps_label_file_name")), order="F")
            sem_seg_gt = pan_seg_gt // 1000
            sem_seg_gt[sem_seg_gt > 18] = self.ignore_label
            sem_seg_gt = sem_seg_gt.astype("double")
        else:
            pan_seg_gt, sem_seg_gt = None, None

        if "depth_label_file_name" in dataset_dict:
            depth_gt = np.array(Image.open(dataset_dict.pop("depth_label_file_name")))
            depth_gt_1 = (depth_gt // 256).astype(np.uint8)
            depth_gt_2 = (depth_gt % 256).astype(np.uint8)
        else:
            depth_gt = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_gt' for DPS dataset {}".format(
                    dataset_dict["file_name"]
                ) 
            )

        # Apply transformations to the image and annotations
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt.astype("double"))

        if depth_gt is not None:
            depth_gt_1 = transforms.apply_segmentation(depth_gt_1)
            depth_gt_2 = transforms.apply_segmentation(depth_gt_2)
            depth_gt = depth_gt_1.astype(np.float64) * 256 + depth_gt_2.astype(np.float64)
            depth_gt = depth_gt / 256.
            del depth_gt_1, depth_gt_2
            for transform in transforms:
                if isinstance(transform, T.ResizeTransform):
                    aug_scale = (transform.w / transform.new_w +
                                 transform.h / transform.new_h) / 2
                    if self.depth_bound:
                        depth_gt = np.clip(depth_gt * aug_scale, depth_gt.min(), depth_gt.max())
                    else:
                        depth_gt = depth_gt * aug_scale
        
        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if depth_gt is not None:
            depth_gt = torch.as_tensor(depth_gt.astype("float32"))
        
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = torch.as_tensor(image.shape[-2:]) + self.size_divisibility - 1
            image_size = image_size.div(self.size_divisibility, rounding_mode="floor") * self.size_divisibility
            image_size = image_size.tolist()
            padding_size = [
                0,
                image_size[1] - image.shape[-1],
                0,
                image_size[0] - image.shape[-2],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            pan_seg_gt = F.pad(pan_seg_gt, padding_size, value=VOID*1000).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            if depth_gt is not None:
                depth_gt = F.pad(depth_gt, padding_size, value=0).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt.long()
        
        if depth_gt is not None:
            depth_gt = depth_gt.float()
            dataset_dict["depth"] = depth_gt

        if "annotations" in dataset_dict:
            raise ValueError("Panoptic segmentation dataset should not have 'annotations'.")

        dataset_dict.pop("vps_label_file_name", None)
        dataset_dict.pop("depth_label_file_name", None)
        dataset_dict.pop("next_frame", None)

        # Prepare per-category binary masks
        indices = torch.unique(pan_seg_gt)
        
        instances = Instances(image_shape)
        classes = []
        masks = []
        if depth_gt is not None:
            depths = []
            mean_depths = []
        for index in indices:
            class_id = index // 1000
            # Ignore invalid segment
            if class_id.item() > 18:
                continue
            classes.append(class_id)
            seg_mask = pan_seg_gt == index
            masks.append(seg_mask)

            if depth_gt is not None:
                seg_depth = torch.zeros_like(depth_gt)
                seg_depth[seg_mask] = depth_gt[seg_mask]
                depths.append(seg_depth)
                valid_seg_depth = seg_depth > 0
                mean_depths.append(seg_depth.sum() / valid_seg_depth.sum().clamp(1))
        
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_classes = torch.zeros(0, dtype=torch.long)
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor
            instances.gt_classes = torch.stack(classes)

        if depth_gt is not None:
            if len(masks) == 0:
                instances.gt_depths = torch.zeros((0, depth_gt.shape[-2], depth_gt.shape[-1]))
                instances.mean_depths = torch.zeros(0)
            else:
                instances.gt_depths = torch.stack(depths)
                instances.mean_depths = torch.stack(mean_depths)         

        dataset_dict["instances"] = instances

        return dataset_dict
