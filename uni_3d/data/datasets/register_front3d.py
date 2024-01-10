import json
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
import torch
"""
This file contains functions to register the 3D-Front dataset to the DatasetCatalog.
"""

FRONT3D_CATEGORIES = [
    {"color": (220, 20, 60), "isthing": 1, "id": 1, "trainId": 1, "name": "cabinet"},
    {"color": (255, 0, 0), "isthing": 1, "id": 2, "trainId": 2, "name": "bed"},
    {"color": (0, 0, 142), "isthing": 1, "id": 3, "trainId": 3, "name": "chair"},
    {"color": (0, 0, 70), "isthing": 1, "id": 4, "trainId": 4, "name": "sofa"},
    {"color": (0, 60, 100), "isthing": 1, "id": 5, "trainId": 5, "name": "table"},
    {"color": (0, 80, 100), "isthing": 1, "id": 6, "trainId": 6, "name": "desk"},
    {"color": (0, 0, 230), "isthing": 1, "id": 7, "trainId": 7, "name": "dresser"},
    {"color": (119, 11, 32), "isthing": 1, "id": 8, "trainId": 8, "name": "lamp"},
    {"color": (190, 50, 60), "isthing": 1, "id": 9, "trainId": 9, "name": "other"},
    {"color": (102, 102, 156), "isthing": 0, "id": 10, "trainId": 10, "name": "wall"},
    {"color": (128, 64, 128), "isthing": 0, "id": 11, "trainId": 11, "name": "floor"},
]

FRONT3D_INTRINSIC = np.array([[277.1281435,   0.       , 159.5,  0.],
                              [  0.       , 277.1281435, 119.5,  0.],
                              [  0.       ,   0.       ,   1. ,  0.],
                              [  0.       ,   0.       ,   0. ,  1.]]).reshape((4, 4))

logger = logging.getLogger(__name__)

_RAW_FRONT_3D_SPLITS = {
    "front3d_train": (
        "front3d/data",
        "front3d/meta/train_3d.json",
        "front3d/meta/frustum_mask.npz",
    ),
    "front3d_val": (
        "front3d/data",
        "front3d/meta/val_3d.json",
        "front3d/meta/frustum_mask.npz",
    ),
}

def load_front3d(image_dir: str, gt_json: str, enable_3d: bool):
    assert os.path.exists(gt_json), gt_json+" not exists"
    with open(gt_json) as f:
        file_dicts = json.load(f)

    ret = []

    for file_dict in file_dicts:
        scene_id, image_id = file_dict["scene_id"], file_dict["image_id"]
        item = {
            "image_id": scene_id + "_" + image_id,
            "file_name": os.path.join(image_dir, scene_id, f"rgb_{image_id}.png"),
            "depth_label_file_name": os.path.join(image_dir, scene_id, f"depth_{image_id}.exr"),
            "segm_label_file_name": os.path.join(image_dir, scene_id, f"segmap_{image_id}.mapped.npz"),
            "height": file_dict["height"],
            "width": file_dict["width"],
            "scene_id": scene_id,
            "raw_image_id": image_id,
        }
        ret.append(item)
        if enable_3d:
            item.update({
                "geometry_file_name": os.path.join(image_dir, scene_id, f"geometry_{image_id}.npz"),
                "segm_3d_file_name": os.path.join(image_dir, scene_id, f"segmentation_{image_id}.mapped.npz"),
                "weighting_file_name": os.path.join(image_dir, scene_id, f"weighting_{image_id}.npz"),
            })
    
    assert len(ret), f"No images found in {image_dir}!"
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))
    return ret

def register_all_front_3d(root):
    meta = {}

    thing_classes = [k["name"] for k in FRONT3D_CATEGORIES]
    thing_colors = [k["color"] for k in FRONT3D_CATEGORIES]
    stuff_classes = [k["name"] for k in FRONT3D_CATEGORIES]
    stuff_colors = [k["color"] for k in FRONT3D_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in FRONT3D_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_json, frustum_mask) in _RAW_FRONT_3D_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_json = os.path.join(root, gt_json)

        for flavor in ("_2d", ""):
            enable_3d = "2d" not in flavor
            if enable_3d:
                meta["intrinsic"] = torch.from_numpy(FRONT3D_INTRINSIC).float()
                frustum_mask_path = os.path.join(root, frustum_mask)
                if os.path.isfile(frustum_mask_path):
                    meta["frustum_mask"] = torch.from_numpy(np.load(frustum_mask_path)["mask"]).bool()
            DatasetCatalog.register(
                key + flavor, lambda x=image_dir, y=gt_json, z=enable_3d: load_front3d(x, y, z)
            )
            MetadataCatalog.get(key + flavor).set(
                image_root=image_dir,
                gt_dir=gt_json,
                evaluator_type="front3d_dps" if not enable_3d else "front3d",
                ignore_label=255,
                label_divisor=1000,
                class_info=FRONT3D_CATEGORIES,
                **meta,
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_front_3d(_root)
