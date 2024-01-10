import json
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog


MATTERPORT_CATEGORIES = [
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
    {"color": (70, 70, 70), "isthing": 0, "id": 12, "trainId": 12, "name": "ceiling"},
]

_RAW_MATTERPORT_SPLITS = {
    "matterport_train": (
        "matterport/data",
        "matterport/meta/train_3d.json",
    ),
    "matterport_val": (
        "matterport/data",
        "matterport/meta/val_3d.json",
    ),
}

logger = logging.getLogger(__name__)


def load_matterport(image_dir: str, gt_json: str, enable_3d: bool):
    assert os.path.exists(gt_json), gt_json+" not exists"
    with open(gt_json) as f:
        file_dicts = json.load(f)

    ret = []

    for file_dict in file_dicts:
        scene_id, image_id = file_dict["scene_id"], file_dict["image_id"]
        name, angle, rot = image_id.split("_")
        item = {
            "image_id": scene_id + "_" + image_id,
            "file_name": os.path.join(image_dir, scene_id, f"{name}_i{angle}_{rot}.jpg"),
            "depth_label_file_name": os.path.join(image_dir, scene_id, f"{name}_d{angle}_{rot}.png"),
            "intrinsic_label_file_name": os.path.join(image_dir, scene_id, f"{name}_intrinsics_{angle}.npy"),
            "segm_label_file_name": os.path.join(image_dir, scene_id, f"{name}_segmap{angle}_{rot}.mapped.npz"),
            "height": file_dict["height"],
            "width": file_dict["width"],
            "scene_id": scene_id,
            "raw_image_id": image_id,
        }
        ret.append(item)
        if enable_3d:
            item.update({
                "geometry_file_name": os.path.join(image_dir, scene_id, f"{name}_geometry{angle}_{rot}.npz"),
                "segm_3d_file_name": os.path.join(image_dir, scene_id, f"{name}_segmentation{angle}_{rot}.mapped.npz"),
                "weighting_file_name": os.path.join(image_dir, scene_id, f"{name}_weighting{angle}_{rot}.npz"),
            })
    
    assert len(ret), f"No images found in {image_dir}!"
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))
    return ret

def register_all_matterport(root):
    meta = {}

    thing_classes = [k["name"] for k in MATTERPORT_CATEGORIES]
    thing_colors = [k["color"] for k in MATTERPORT_CATEGORIES]
    stuff_classes = [k["name"] for k in MATTERPORT_CATEGORIES]
    stuff_colors = [k["color"] for k in MATTERPORT_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in MATTERPORT_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_json) in _RAW_MATTERPORT_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_json = os.path.join(root, gt_json)

        for flavor in ("_2d", ""):
            enable_3d = "2d" not in flavor
            DatasetCatalog.register(
                key + flavor, lambda x=image_dir, y=gt_json, z=enable_3d: load_matterport(x, y, z)
            )
            MetadataCatalog.get(key + flavor).set(
                image_root=image_dir,
                gt_dir=gt_json,
                evaluator_type="matterport_dps" if not enable_3d else "front3d",
                ignore_label=255,
                label_divisor=1000,
                class_info=MATTERPORT_CATEGORIES,
                **meta,
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_matterport(_root)
