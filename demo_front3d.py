from typing import Dict, Any
import argparse
import json
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

from uni_3d import add_mask2former_config, add_uni_3d_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
import uni_3d.utils.visualize as vis
import uni_3d.utils.mesh as mesh
from uni_3d.modeling.reconstruction.frustum import generate_frustum, compute_camera2frustum_transform


def main(args):
    cfg = setup_inference(args)

    input_format = cfg.INPUT.FORMAT
    assert input_format in ["RGB", "BGR"], input_format

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    device = torch.device(args.device)
    # Define model and load checkpoint.
    print("Loading model...")
    model = build_model(cfg).to(device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    im = cv2.imread(args.input)
    if input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        im = im[:, :, ::-1]
    height, width = im.shape[:2]
    image = aug.get_transform(im).apply_image(im)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    intrinsic = metadata.intrinsic
    frustum_mask = metadata.frustum_mask

    inputs = {"image": image, "height": height, "width": width, 
              "frustum_mask": frustum_mask, "intrinsic": intrinsic}

    print("Perform panoptic 3D scene reconstruction...")
    with torch.no_grad():
        results = model([inputs])[0]

    print(f"Visualize results, save them at {cfg.OUTPUT_DIR}")
    visualize_results(cfg, im, results)


def setup_inference(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask2former_config(cfg)
    add_uni_3d_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output
    cfg.MODEL.WEIGHTS = args.model
    cfg.freeze()
    return cfg


def to_pointcloud(intrinsic, depth, image_size, filename):
    pointcloud, _ = compute_pointcloud(intrinsic, depth, image_size)
    mesh.write_pointcloud(pointcloud, None, filename)


def to_pointcloud_with_colors(intrinsic, depth, image_size, colors, filename):
    pointcloud, coords = compute_pointcloud(intrinsic, depth, image_size)
    color_values = colors[coords[:, 0], coords[:, 1]]
    mesh.write_pointcloud(pointcloud, color_values, filename)


def compute_pointcloud(intrinsic, depth, image_size):
    depth_pixels_xy = depth.nonzero(as_tuple=False)
    device = depth_pixels_xy.device
    intrinsic = intrinsic.to(device)
    depth_pixels_z = depth[depth_pixels_xy[:, 0], depth_pixels_xy[:, 1]].reshape(-1).float()
    depth_pixels_xy = depth_pixels_xy.flip(-1).float()
    normalized_depth_pixels_xy = depth_pixels_xy / torch.tensor([depth.shape[-1], depth.shape[-2]], device=device)
    xv, yv = (normalized_depth_pixels_xy * torch.tensor(image_size, device=device) * depth_pixels_z[:, None]).unbind(-1)

    depth_pixels = torch.stack([xv, yv, depth_pixels_z, torch.ones_like(depth_pixels_z)])
    pointcloud = torch.mm(torch.inverse(intrinsic), depth_pixels.float()).t()[:, :3]

    return pointcloud, depth_pixels_xy


def visualize_results(cfg, image, results: Dict[str, Any]) -> None:
    output_path = Path(cfg.OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)

    iso_value = 1.25
    depth_min = cfg.MODEL.UNI_3D.PROJECTION.DEPTH_MIN
    depth_max = cfg.MODEL.UNI_3D.PROJECTION.DEPTH_MAX
    frustum_dims = cfg.MODEL.UNI_3D.FRUSTUM3D.GRID_DIMENSIONS


    surface = results["geometry"].squeeze()
    instances = results["panoptic_seg"].squeeze()
    semantics = results["semantic_seg"].squeeze()
    mapping = results["panoptic_semantic_mapping"]
    
    r_instances = torch.zeros_like(instances)
    cid = 4
    for sid, cls_id in mapping.items():
        if cls_id == 10:
            r_instances[instances == sid] = 1
        elif cls_id == 11:
            r_instances[instances == sid] = 2
        elif cls_id == 12:
            r_instances[instances == sid] = 3
        else:
            r_instances[instances == sid] = cid
            cid += 1
    instances = r_instances

    vis.write_image(image, output_path / "input_image.png")
    
    # Visualize depth prediction
    to_pointcloud(results["intrinsic"], results["depth"], results["image_size"], output_path / "depth_prediction.ply")
    vis.write_depth(
        F.interpolate(results["depth"][None, None], results["image_size"][::-1]).squeeze(),
        output_path / "depth_map.png"
    )

    # Visualize 2D segmentation
    vis.write_segmentation_image(image, results["panoptic_seg_2d"], output_path / "segmentation.png")

    # Visualize projection
    # mesh.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs

    # Main outputs
    frustum = generate_frustum(results["image_size"], torch.inverse(results["intrinsic"].cpu()), depth_min, depth_max)
    camera2frustum, padding_offsets = compute_camera2frustum_transform(frustum, 
                                                                       cfg.MODEL.UNI_3D.PROJECTION.VOXEL_SIZE,
                                                                       torch.tensor([frustum_dims] * 3))


    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += padding_offsets
    frustum2camera = torch.inverse(camera2frustum)
    print(frustum2camera)
    mesh.write_distance_field(surface, None, output_path / "mesh_geometry.ply", transform=frustum2camera, iso_value=iso_value)
    mesh.write_distance_field(surface, instances, output_path / "mesh_instances.ply", transform=frustum2camera, iso_value=iso_value)
    mesh.write_distance_field(surface, semantics, output_path / "mesh_semantics.ply", transform=frustum2camera, iso_value=iso_value)

    with open(output_path / "semantic_classes.json", "w") as f:
        json.dump(results["panoptic_semantic_mapping"], f, indent=4)

    surface_mask = surface < iso_value
    points = surface_mask.nonzero()
    point_semantics = semantics[surface_mask]
    point_instances = instances[surface_mask]

    mesh.write_pointcloud(points, None, output_path / "points_geometry.ply")
    mesh.write_semantic_pointcloud(points, point_semantics, output_path / "points_surface_semantics.ply")
    mesh.write_semantic_pointcloud(points, point_instances, output_path / "points_surface_instances.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="Path to an image from 3D-Front dataset", default="figures/demo.png")
    parser.add_argument("--output", "-o", type=str, help="Output path", default="output/demo")
    parser.add_argument("--config-file", "-c", type=str, help="Path to config file", default="configs/front3d/uni_3d_R50.yaml")
    parser.add_argument("--model", "-m", type=str, help="Path to pre-trained model weight", default="models/front3d_full_single_scale.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    main(args)
