import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
import MinkowskiEngine as Me
from uni_3d import MaskFormer
from uni_3d.modeling.reconstruction import SparseProjection, FrustumDecoder
from uni_3d.evaluation.utils import prepare_instance_masks_thicken
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils.sparse_tensor import to_dense


@META_ARCH_REGISTRY.register()
class Uni3D(MaskFormer):
    @configurable
    def __init__(
        self,
        *args,
        truncation,
        frustum_dims,
        reprojection,
        completion,
        iso_value,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.depth_on, "Depth estimation must be enabled for Uni3D"

        # Disable gradients for the 2D model
        for _, param in self.named_parameters():
            param.requires_grad_(False)
        
        # 2D to 3D models
        self.reprojection = reprojection
        self.completion = completion
        self.truncation = truncation
        self.frustum_dims = [frustum_dims] * 3
        self.iso_value = iso_value

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # depth_weight = cfg.MODEL.MASK_FORMER.DEPTH_WEIGHT
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        panoptic_weight = cfg.MODEL.UNI_3D.FRUSTUM3D.PANOPTIC_WEIGHT
        occupancy_weights = cfg.MODEL.UNI_3D.FRUSTUM3D.COMPLETION_WEIGHTS
        geometry_weight = cfg.MODEL.UNI_3D.FRUSTUM3D.SURFACE_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_geometry": geometry_weight,
        }

        for idx, lvl in enumerate([64, 128, 256]):
            weight_dict[f"loss_occupancy_{lvl}"] = occupancy_weights[idx]
            weight_dict[f"loss_panoptic_{lvl}"] = panoptic_weight

        losses = ["geometry", "occupancy", "panoptic"]

        criterion = SetCriterion(
            ret["sem_seg_head"].num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            depth_scale=cfg.MODEL.MASK_FORMER.DEPTH_SCALE,
        )

        ret["reprojection"] = SparseProjection(cfg)
        ret["completion"] = FrustumDecoder(cfg)
        ret["criterion"] = criterion
        ret["truncation"] = cfg.MODEL.UNI_3D.FRUSTUM3D.TRUNCATION
        ret["frustum_dims"] = cfg.MODEL.UNI_3D.FRUSTUM3D.GRID_DIMENSIONS
        ret["iso_value"] = cfg.MODEL.UNI_3D.FRUSTUM3D.ISO_VALUE
        return ret
    
    def forward(self, batched_inputs):
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            if "room_mask" in batched_inputs[0]:
                room_mask = torch.stack([x["room_mask"] for x in batched_inputs]).to(self.device)
            else:
                room_mask = None

            # 2D inference
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            depth_pred_results = outputs["pred_depths"] * self.depth_scale

            padded_out_h, padded_out_w = images.tensor.shape[-2] // 2, images.tensor.shape[-1] // 2
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(padded_out_h, padded_out_w),
                mode="bilinear",
                align_corners=False,
            )
            depth_pred_results = F.interpolate(
                depth_pred_results,
                size=(padded_out_h, padded_out_w),
                mode="bilinear",
                align_corners=False,
            )
            processed_results = []
            for mask_cls_result, mask_pred_result, depth_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, depth_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                out_h, out_w = height // 2, width // 2
                processed_results.append({})
                # Remove padding due to size divisibility
                mask_pred_result = mask_pred_result[:, :out_h, :out_w]
                depth_pred_result = depth_pred_result[:, :out_h, :out_w]
                
                panoptic_seg, depth_r, segments_info, sem_prob_masks = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result, depth_pred_result)

                if room_mask is not None:
                    idx = len(processed_results) - 1
                    mask = F.interpolate(
                        room_mask[[idx], None].float(), 
                        size=depth_r.shape,
                        mode="nearest",
                    )[0, 0].bool()

                    depth_r[~mask] = 0

                processed_results[-1]["panoptic_seg"] = (panoptic_seg, segments_info)
                processed_results[-1]["depth"] = depth_r
                processed_results[-1]["image_size"] = (width, height)
                processed_results[-1]["padded_size"] = (images.tensor.shape[-1], images.tensor.shape[-2])
                processed_results[-1]["intrinsic"] = input_per_image["intrinsic"]
                processed_results[-1]["sem_seg"] = sem_prob_masks

        multi_scale_features = list(reversed(outputs["enc_features"]))
        encoder_features = torch.cat([outputs["mask_features"], outputs["depth_features"]], dim=1)
        sparse_multi_scale_features, sparse_encoder_features = self.reprojection(multi_scale_features, encoder_features, processed_results)

        ## 3D frustum completion
        segm_queries = outputs["segm_decoder_out"]
        frustum_mask = torch.stack([x["frustum_mask"] for x in batched_inputs]).to(self.device)
        frustum_mask_64 = F.max_pool3d(frustum_mask[:, None].float(), kernel_size=2, stride=4).bool()

        outputs_3d = self.completion(sparse_multi_scale_features, sparse_encoder_features, segm_queries, frustum_mask_64)

        if self.training:            
            # Copy over 2D results for matching
            outputs_3d["pred_logits"] = outputs["pred_logits"]
            outputs_3d["pred_masks"] = outputs["pred_masks"]

            if "instances" in batched_inputs[0]:
                targets = self.prepare_targets(batched_inputs, images)
            else:
                targets = None
            
            losses = self.criterion(outputs_3d, targets)
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            dense_dimensions = torch.Size([1, 1] + self.frustum_dims)
            min_coordinates = torch.IntTensor([0, 0, 0])
            
            geometry_results = to_dense(outputs_3d["pred_geometry"], dense_dimensions, min_coordinates, default_value=self.truncation)[0]
            mask_3d_results = outputs_3d["pred_segms"][-1]
            
            mask_cls_results = outputs["pred_logits"]

            processed_results_3d = []

            for idx, (geometry_result, mask_cls_result) in enumerate(zip(geometry_results, mask_cls_results)):
                coords, mask_3d = mask_3d_results.coordinates_at(idx), mask_3d_results.features_at(idx)
                coords, mask_3d = Me.utils.sparse_collate([coords], [mask_3d])
                geometry_result = geometry_result.squeeze(0)
                panoptic_seg, panoptic_semantic_mapping, semantic_seg = self.panoptic_3d_inference(
                    geometry_result, mask_cls_result, (coords, mask_3d, mask_3d_results.tensor_stride), min_coordinates, dense_dimensions,
                )
                downsample_factor = batched_inputs[idx]["downsample_factor"]
                processed_results_3d.append({
                    "intrinsic": processed_results[idx]["intrinsic"],
                    "image_size": processed_results[idx]["image_size"],
                    "depth": processed_results[idx]["depth"],
                    "panoptic_seg_2d": processed_results[idx]["panoptic_seg"],
                    "geometry": geometry_result,
                    "panoptic_seg": panoptic_seg,
                    "semantic_seg": semantic_seg,
                    "panoptic_semantic_mapping": panoptic_semantic_mapping,
                    "instance_info_pred": prepare_instance_masks_thicken(panoptic_seg, 
                                                                         panoptic_semantic_mapping, 
                                                                         geometry_result, 
                                                                         frustum_mask[idx],
                                                                         iso_value=self.iso_value,
                                                                         downsample_factor=downsample_factor)
                })

            return processed_results_3d

    def panoptic_3d_inference(self, geometry, mask_cls, sparse_mask_tuple, min_coordinates, dense_dimensions):
        panoptic_seg = torch.zeros(geometry.shape, dtype=torch.int32, device=mask_cls.device)
        semantic_seg = torch.zeros_like(panoptic_seg)
        panoptic_semantic_mapping = {}

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)

        keep = labels.ne(self.sem_seg_head.num_classes) & \
               labels.ne(0) & \
               (scores > self.object_mask_threshold)
        
        coords, sparse_masks, stride = sparse_mask_tuple

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = Me.MinkowskiSigmoid()(
            Me.SparseTensor(features=sparse_masks[:, keep], coordinates=coords, tensor_stride=stride)
        ).dense(dense_dimensions, min_coordinates)[0].squeeze(0)
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks

        current_segment_id = 0

        if cur_masks.shape[0] > 0:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            query_to_segment_id = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                
                if mask.sum().item() > 0:
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            query_to_segment_id[k] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    query_to_segment_id[k] = current_segment_id
                    panoptic_semantic_mapping[current_segment_id] = int(pred_class)

            surface_mask = geometry.abs() <= 1.5

            # Fill unassigned surface voxels
            unassigned_mask = surface_mask & (panoptic_seg == 0)
            for k in range(cur_classes.shape[0]):
                mask = (cur_mask_ids == k) & unassigned_mask
                if mask.sum().item() > 0 and k in query_to_segment_id.keys():
                    panoptic_seg[mask] = query_to_segment_id[k]
            
            for segm_id, semantic_label in panoptic_semantic_mapping.items():
                instance_mask = panoptic_seg == segm_id
                semantic_seg[instance_mask] = semantic_label

        return panoptic_seg, panoptic_semantic_mapping, semantic_seg
    
    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for x in batched_inputs:
            inst_per_image = x["instances"].to(self.device)
            # pad gt
            gt_masks = inst_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            gt_depths = inst_per_image.gt_depths
            padded_depths = torch.zeros((gt_depths.shape[0], h_pad, w_pad), dtype=gt_depths.dtype, device=gt_depths.device)
            padded_depths[:, : gt_depths.shape[1], : gt_depths.shape[2]] = gt_depths
            
            target = {
                "labels": inst_per_image.gt_classes,
                "masks": padded_masks,
                "depths": padded_depths,
                "masks_3d_256": inst_per_image.gt_masks_3d_256,
                "masks_3d_128": inst_per_image.gt_masks_3d_128,
                "masks_3d_64": inst_per_image.gt_masks_3d_64,
                "occupancy_256": x["occupancy_256"].to(self.device),
                "occupancy_128": x["occupancy_128"].to(self.device),
                "occupancy_64": x["occupancy_64"].to(self.device),
                "weighting3d_256": x["weighting3d_256"].to(self.device),
                "weighting3d_128": x["weighting3d_128"].to(self.device),
                "weighting3d_64": x["weighting3d_64"].to(self.device),
                "geometry": x["geometry"].to(self.device),
            }
            new_targets.append(target)
        return new_targets
