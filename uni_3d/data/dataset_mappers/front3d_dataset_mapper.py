from typing import Dict
import copy
import numpy as np
import torch
from torch.nn import functional as F
import numpy as np
import pyexr

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from uni_3d.evaluation.utils import prepare_instance_masks_thicken
import uni_3d.data.transforms3d as t3d
from .cityscapes_dps_dataset_mapper import CityscapesDPSDatasetMapper

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["Front3DDPSDatasetMapper", "Front3DDatasetMapper"]


STUFF_CLASSES = [10, 11, 12]


class Front3DDPSDatasetMapper(CityscapesDPSDatasetMapper):
    def __call__(self, dataset_dict):
        assert self.is_train, "Front3DDPSDatasetMapper should only be used for training!"

        depth_min = self.config.MODEL.UNI_3D.PROJECTION.DEPTH_MIN
        depth_max = self.config.MODEL.UNI_3D.PROJECTION.DEPTH_MAX

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        is_matterport = "matterport" in self.dataset_name

        if "segm_label_file_name" in dataset_dict:
            pan_seg_gt = np.load(dataset_dict.pop("segm_label_file_name"))["data"]
            sem_seg_gt, inst_seg_gt = pan_seg_gt[..., 0], pan_seg_gt[..., 1]
            del pan_seg_gt
            sem_seg_gt[sem_seg_gt == 0] = self.ignore_label
            sem_seg_gt = sem_seg_gt.astype("double")
            inst_seg_gt = inst_seg_gt.astype("double")
        else:
            sem_seg_gt, inst_seg_gt = None, None

        if "room_mask_file_name" in dataset_dict:
            room_mask = utils.read_image(dataset_dict.pop("room_mask_file_name"))
            room_mask = room_mask.astype("double")
        else:
            room_mask = None

        if "depth_label_file_name" in dataset_dict:
            if is_matterport:
                depth_gt = utils.read_image(dataset_dict.pop("depth_label_file_name")).astype("double") / 4000
                depth_gt[depth_gt < depth_min] = 0
                depth_gt[depth_gt > depth_max] = 0
            else:
                depth_gt = pyexr.read(dataset_dict.pop("depth_label_file_name")).squeeze().copy().astype("double")
        else:
            depth_gt = None

        if sem_seg_gt is None or inst_seg_gt is None:
            raise ValueError("Panoptic segmentation not present in dataset")

        # Apply transformations to the image and annotations
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        inst_seg_gt = transforms.apply_segmentation(inst_seg_gt)
        
        if room_mask is not None:
            room_mask = transforms.apply_segmentation(room_mask)
            room_mask = torch.as_tensor(room_mask.astype("long"))

        if depth_gt is not None:
            depth_gt = transforms.apply_segmentation(depth_gt)
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
        
        inst_seg_gt = torch.as_tensor(inst_seg_gt.astype("long"))
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
            inst_seg_gt = F.pad(inst_seg_gt, padding_size, value=-1).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            if room_mask is not None:
                room_mask = F.pad(room_mask, padding_size, value=0).contiguous()
            if depth_gt is not None:
                depth_gt = F.pad(depth_gt, padding_size, value=0).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt.long()

        if room_mask is not None:
            dataset_dict["room_mask"] = room_mask > 0
        
        if depth_gt is not None:
            depth_gt = depth_gt.float()
            dataset_dict["depth"] = depth_gt

        if "annotations" in dataset_dict:
            raise ValueError("Panoptic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        indices = torch.unique(inst_seg_gt)
        
        instances = Instances(image_shape)
        classes = []
        masks = []
        if depth_gt is not None:
            depths = []
            mean_depths = []

        def _add_results(_class_id, _seg_mask):
            classes.append(_class_id)
            masks.append(_seg_mask)

            if depth_gt is not None:
                seg_depth = torch.zeros_like(depth_gt)
                seg_depth[_seg_mask] = depth_gt[_seg_mask]
                depths.append(seg_depth)
                valid_seg_depth = seg_depth > 0
                mean_depths.append(seg_depth.sum() / valid_seg_depth.sum().clamp(1))
        
        inst_ids = []
        for index in indices:
            if index <= 0:
                continue

            seg_mask = inst_seg_gt == index

            if seg_mask.sum() <= self.min_instance_pixels:
                continue

            # Determine semantic label of the current instance by voting
            semantic_labels = sem_seg_gt[seg_mask]
            unique_semantic_labels, semantic_label_count = torch.unique(semantic_labels, return_counts=True)
            max_semantic_label = torch.argmax(semantic_label_count)
            class_id = unique_semantic_labels[max_semantic_label]

            if class_id == self.ignore_label or class_id in STUFF_CLASSES:
                continue
            
            inst_ids.append(index)
            _add_results(class_id, seg_mask)

        stuff_ids = []
        for class_id in torch.as_tensor(STUFF_CLASSES):
            seg_mask = sem_seg_gt == class_id
            if seg_mask.sum() == 0:
                continue

            stuff_ids.append(class_id)
            _add_results(class_id, seg_mask)
            
        
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, inst_seg_gt.shape[-2], inst_seg_gt.shape[-1]))
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
        dataset_dict["inst_ids"] = torch.stack(inst_ids) if len(inst_ids) else torch.empty(0)
        dataset_dict["stuff_ids"] = torch.stack(stuff_ids) if len(stuff_ids) else torch.empty(0)

        return dataset_dict


class Front3DDatasetMapper(Front3DDPSDatasetMapper):
    @configurable
    def __init__(self, 
                 metadata,
                 truncation: float,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        self.truncation = truncation
        self.is_matterport = "matterport" in self.dataset_name
        self.transforms = self.define_transformations()

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret["truncation"] = cfg.MODEL.UNI_3D.FRUSTUM3D.TRUNCATION

        dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])

        ret["metadata"] = meta
        return ret

    def define_transformations(self) -> Dict:
        transforms = {}
        
        transforms["geometry"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.float),
            t3d.ToTDF(truncation=12)
        ])
        transforms["geometry_truncate"] = t3d.ToTDF(truncation=self.truncation)

        transforms["occupancy_64"] = t3d.Compose([t3d.ResizeTrilinear(0.25), t3d.ToBinaryMask(8), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_128"] = t3d.Compose([t3d.ResizeTrilinear(0.5), t3d.ToBinaryMask(6), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_256"] = t3d.Compose([t3d.ToBinaryMask(self.truncation), t3d.ToTensor(dtype=torch.float)])

        transforms["weighting3d"] = t3d.ToTensor(dtype=torch.float)
        transforms["weighting3d_64"] = t3d.ResizeTrilinear(0.25)
        transforms["weighting3d_128"] = t3d.ResizeTrilinear(0.5)

        transforms["semantic3d"] = t3d.ToTensor(dtype=torch.long)

        transforms["segmentation3d_64"] = t3d.ResizeMax(8, 4, 2)
        transforms["segmentation3d_128"] = t3d.ResizeMax(4, 2, 1)

        return transforms

        
    def __call__(self, dataset_dict):
        assert self.is_train, "Front3DDatasetMapper should only be used for training!"
        dataset_dict: dict = super().__call__(dataset_dict)

        ## Load geometry
        geometry_content = np.load(dataset_dict.pop("geometry_file_name"))
        geometry = geometry_content["data"]
        if not self.is_matterport:
            geometry = np.ascontiguousarray(np.flip(geometry, axis=[0, 1]))
        
        geometry = self.transforms["geometry"](geometry)
        dataset_dict["occupancy_256"] = self.transforms["occupancy_256"](geometry)
        dataset_dict["occupancy_128"] = self.transforms["occupancy_128"](geometry)
        dataset_dict["occupancy_64"] = self.transforms["occupancy_64"](geometry)

        geometry = self.transforms["geometry_truncate"](geometry)
        dataset_dict["geometry"] = geometry

        ## Panoptic
        segm_3d = np.load(dataset_dict.pop("segm_3d_file_name"))["data"]
        if not self.is_matterport:
            segm_3d = np.copy(np.flip(segm_3d, axis=[1, 2]))
        
        semantic3d, instance3d = segm_3d
        semantic3d = self.transforms["semantic3d"](semantic3d)
        instance3d = self.transforms["semantic3d"](instance3d)

        if self.is_matterport:
            # The frustum mask for matterport is inverted (True is masked)
            dataset_dict["frustum_mask"] = ~torch.as_tensor(geometry_content["mask"], dtype=torch.bool)
            dataset_dict["intrinsic"] = torch.from_numpy(np.load(dataset_dict.pop("intrinsic_label_file_name")).reshape(4, 4)).float()
        else:
            dataset_dict["frustum_mask"] = self.metadata.frustum_mask
            dataset_dict["intrinsic"] = self.metadata.intrinsic

        segm_3d_masks = []
        for index in dataset_dict["inst_ids"]:
            segm_3d_masks.append(instance3d == index)
        
        for class_id in dataset_dict["stuff_ids"]:
            segm_3d_masks.append(semantic3d == class_id)

        if len(segm_3d_masks) > 0:
            segm_3d_masks = torch.stack(segm_3d_masks)
            dataset_dict["instances"].gt_masks_3d_256 = segm_3d_masks
            dataset_dict["instances"].gt_masks_3d_128 = self.transforms["segmentation3d_128"](segm_3d_masks)
            dataset_dict["instances"].gt_masks_3d_64 = self.transforms["segmentation3d_64"](segm_3d_masks)
        else:
            dataset_dict["instances"].gt_masks_3d_256 = torch.zeros((0,) + dataset_dict["occupancy_256"].shape)
            dataset_dict["instances"].gt_masks_3d_128 = torch.zeros((0,) + dataset_dict["occupancy_128"].shape)
            dataset_dict["instances"].gt_masks_3d_64 = torch.zeros((0,) + dataset_dict["occupancy_64"].shape)

        ## Weighting
        weighting = np.load(dataset_dict.pop("weighting_file_name"))["data"]
        if not self.is_matterport:
            weighting = np.copy(np.flip(weighting, axis=[0, 1]))
        weighting = self.transforms["weighting3d"](weighting)
        dataset_dict["weighting3d_256"] = weighting
        dataset_dict["weighting3d_128"] = self.transforms["weighting3d_128"](weighting)
        dataset_dict["weighting3d_64"] = self.transforms["weighting3d_64"](weighting)

        return dataset_dict


class Front3DTestDatasetMapper(Front3DDatasetMapper):
    @classmethod
    def from_config(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST, 
                cfg.INPUT.MAX_SIZE_TEST, 
                "choice",
            )
        ]

        dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": False,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "depth_bound": cfg.INPUT.DEPTH_BOUND,
            "truncation": cfg.MODEL.UNI_3D.FRUSTUM3D.TRUNCATION,
            "metadata": meta,
            "config": cfg,
            "dataset_name": dataset_names[0],
        }

        return ret

    @staticmethod
    def _prepare_semantic_mapping(instances, semantics, stuff_classes):
        semantic_mapping = {}
        panoptic_instances = torch.zeros_like(instances).int()

        things_start_index = len(stuff_classes)  # map wall and floor to id 1 and 2

        non_thing_classes = torch.tensor([0,] + stuff_classes, device=instances.device)

        unique_instances = instances.unique()
        for index, instance_id in enumerate(unique_instances):
            # Ignore freespace
            if instance_id != 0:
                # Compute 3d instance surface mask
                instance_mask: torch.Tensor = (instances == instance_id)
                # instance_surface_mask = instance_mask & surface_mask
                panoptic_instance_id = index + things_start_index
                panoptic_instances[instance_mask] = panoptic_instance_id

                # get semantic prediction
                semantic_region = torch.masked_select(semantics, instance_mask)
                thing_mask = torch.isin(semantic_region, non_thing_classes, invert=True)
                if (thing_mask.sum() == 0):
                    continue
                semantic_things = semantic_region[thing_mask]

                unique_labels, semantic_counts = torch.unique(semantic_things, return_counts=True)
                max_count, max_count_index = torch.max(semantic_counts, dim=0)
                selected_label = unique_labels[max_count_index]

                semantic_mapping[panoptic_instance_id] = selected_label.int().item()

        # Merge stuff classes
        for idx, stuff_class in enumerate(stuff_classes):
            stuff_mask = semantics == stuff_class
            if (stuff_mask.sum() == 0):
                continue
            stuff_id = idx + 1
            panoptic_instances[stuff_mask] = stuff_id
            semantic_mapping[stuff_id] = stuff_class

        return panoptic_instances, semantic_mapping
    
    
    def __call__(self, dataset_dict):        
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        # Apply transformations to the image and annotations
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, T.AugInput(image))
        image = aug_input.image
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        iso_value = self.config.MODEL.UNI_3D.FRUSTUM3D.ISO_VALUE

        if self.is_matterport:
            room_mask = utils.read_image(dataset_dict.pop("room_mask_file_name"))
            room_mask = room_mask.astype("double")
            room_mask = transforms.apply_segmentation(room_mask)
            room_mask = torch.as_tensor(room_mask.astype("long"))

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

            if self.is_matterport:
                room_mask = F.pad(room_mask, padding_size, value=0).contiguous()

        dataset_dict["image"] = image
        if self.is_matterport:
            dataset_dict["room_mask"] = room_mask > 0

        geometry_content = np.load(dataset_dict.pop("geometry_file_name"))
        geometry = geometry_content["data"]
        if not self.is_matterport:
            geometry = np.ascontiguousarray(np.flip(geometry, axis=[0, 1]))
        geometry = self.transforms["geometry"](geometry)
        dataset_dict["geometry"] = self.transforms["geometry_truncate"](geometry)

        segm_3d = np.load(dataset_dict.pop("segm_3d_file_name"))["data"]
        if not self.is_matterport:
            segm_3d = np.copy(np.flip(segm_3d, axis=[1, 2]))
        semantic3d, instance3d = segm_3d
        semantic3d = self.transforms["semantic3d"](semantic3d)
        instance3d = self.transforms["semantic3d"](instance3d)

        if self.is_matterport:
            stuff_classes = [10, 11, 12]
        else:
            stuff_classes = [10, 11]
        instances_gt, instance_semantic_classes_gt = self._prepare_semantic_mapping(instance3d, semantic3d, stuff_classes=stuff_classes)

        if self.is_matterport:
            # The frustum mask for matterport is inverted (True is masked)
            dataset_dict["frustum_mask"] = ~torch.as_tensor(geometry_content["mask"], dtype=torch.bool)
            dataset_dict["intrinsic"] = torch.from_numpy(np.load(dataset_dict.pop("intrinsic_label_file_name")).reshape(4, 4)).float()
        else:
            dataset_dict["frustum_mask"] = self.metadata.frustum_mask
            dataset_dict["intrinsic"] = self.metadata.intrinsic

        downsample_factor = 2 if self.is_matterport else 1
        instance_information_gt = prepare_instance_masks_thicken(instances_gt, instance_semantic_classes_gt,
                                                                 geometry, dataset_dict["frustum_mask"], 
                                                                 iso_value=iso_value, downsample_factor=downsample_factor)
        
        dataset_dict["instance_info_gt"] = instance_information_gt
        dataset_dict["downsample_factor"] = downsample_factor
        
        return dataset_dict
