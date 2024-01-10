import os
from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .mesh import lookup_colors


def write_image(image: Union[np.array, torch.Tensor], output_file: os.PathLike, **kwargs) -> None:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    plt.imsave(output_file, image, **kwargs)


def write_segmentation_image(image: Union[np.array, torch.Tensor], pred_segs, output_file: os.PathLike) -> None:
    panoptic_seg, segments_info = pred_segs
    # bounding_box = detections.bbox
    # label = detections.get_field("label")
    # masks = detections.get_field("mask2d")

    panoptic_seg = F.interpolate(panoptic_seg[None, None].float(), image.shape[:-1], mode="nearest").squeeze().long().cpu().numpy()
    id_to_cls = {k["id"]: k["category_id"] for k in segments_info}

    masked_image = image.astype(np.uint8).copy()
    seg_ids = np.unique(panoptic_seg)

    for idx in seg_ids:
        if idx == 0:
            continue
        color = lookup_colors(np.array(idx))

        masked_image = apply_mask(masked_image, panoptic_seg==idx, color, alpha=0.8)

    write_image(masked_image, output_file)


def apply_mask(image: np.array, mask: np.array, color, alpha=0.5):
    image[mask] = (image[mask] * (1 - alpha) + alpha * color).astype(np.uint8)
    return image


def write_rgb_image(image: Union[np.array, torch.Tensor], output_file: os.PathLike) -> None:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    write_image(image, output_file)


def write_depth(depth_map: Union[np.array, torch.Tensor], output_file: os.PathLike) -> None:
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()

    write_image(depth_map, output_file, cmap="inferno", vmin=1.0, vmax=6.0)