# Uni-3D: A Universal Model for Panoptic 3D Scene Reconstruction

[Xiang Zhang*](https://xzhang.dev), [Zeyuan Chen*](https://zeyuan-chen.com), [Fangyin Wei](https://weify627.github.io), and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/) (\*Equal contribution)

This is the repository for the paper [Uni-3D: A Universal Model for Panoptic 3D Scene Reconstruction](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Uni-3D_A_Universal_Model_for_Panoptic_3D_Scene_Reconstruction_ICCV_2023_paper.pdf) (ICCV 2023).

[[`Paper`](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Uni-3D_A_Universal_Model_for_Panoptic_3D_Scene_Reconstruction_ICCV_2023_paper.pdf)]

<img src="figures/main.png" />

## Getting Started

### Environment Setup

#### (Recommended) Docker Image
We have pre-packaged all dependencies via Docker [image](https://hub.docker.com/r/zx1239856/uni-3d/tags). It is built on top of `PyTorch 2.1.2` with `CUDA 11.8`.

You can pull the image via
```
docker pull zx1239856/uni-3d:0.1.0
```

#### Manual Approach

Assume you already have proper `PyTorch (>=1.10.1)` and `CUDA (>=11.3)` installation.

1. Install the following system dependencies
```bash
apt-get install ninja-build libopenblas-dev libopenexr-dev
```

2. Remove the comment mark on Line 9 of [requirements.txt](requirements.txt). Install the required Python packages via
```bash
pip install -r requirements.txt
```

### Dataset Preparation

#### 3D-FRONT

Please download [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) from [Dahnert et al. (Panoptic 3D Scene Reconstruction from a Single RGB Image)](https://github.com/xheon/panoptic-reconstruction/tree/main#download). Extract it under `datasets/front3d/data` as 
```
unzip front3d.zip -d datasets/front3d/data
```

#### Matterport3D

Please request the dataset from the authors of [Pano-Re](https://github.com/xheon/panoptic-reconstruction). Extract it under `datasets/matterport/data`. 

Also download the room mask and depth from [BUOL](https://github.com/chtsy/buol). Extract them under`dataset/matterport/room_mask` and `dataset/matterport/depth_gen`, respectively.

##### Folder Structure
```
matterport/
    meta/
        train_3d.json                                         # Training set metadata
        ...
    data/
        <scene_id>/            
            ├── <image_id>_i<frame_id>.png                    # Color image: 320x240x3
            ├── <image_id>_segmap<frame_id>.mapped.npz        # 2D Segmentation: 320x240x2, with 0: pre-mapped semantics, 1: instances
            ├── <image_id>_intrinsics_<camera_id>.png         # Intrinsics matrix: 4x4
            ├── <image_id>_geometry<frame_id>.npz             # 3D Geometry: 256x256x256x1, truncated, (unsigned) distance field at 3cm voxel resolution and 12 voxel truncation.
            ├── <image_id>_segmentation<frame_id>.mapped.npz  # 3D Segmentation: 256x256x256x2, with 0: pre-mapped semantics & instances
            ├── <image_id>_weighting<frame_id>.npz            # 3D Weighting mask: 256x256x256x1
    depth_gen/
        <scene_id>/     
            ├── <posithion_id>_d<frame_id>.png                # Depth image: 320x240x1
    room_mask/
        <scene_id>/   
            ├── <posithion_id>_rm<frame_id>.png               # Room mask: 320x240x1
```

### Pre-trained Weights

| Model                    | PRQ  | RSQ  | RRQ  | Download |
| ------------------------ | :--: | :--: | :--: | -------- |
| 3D-FRONT Pretrained 2D   |  --  |  --  |  --  | [front3d_dps_160k.pth](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/EboqJU7ZZ2FCiWxKKd_9BMgB2AGUxGO8DbGlo7r95GCAoA?e=s6Ok8e) |
| 3D-FRONT Single-scale    | 52.51 | 60.89 | 83.97 | [front3d_full_single_scale.pth](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/EeElQbNMin9IohabPpGrMFUBmpDpeozXfpgy1Fj2h1ZS6w?e=AzaruR) |
| 3D-FRONT Multi-scale     | 53.53 | 61.69 | 84.69 | [front3d_full_multi_scale.pth](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/EcnDpfw1ZdJOgf-sNRHRtZ4B-OvwMH-ldS-h3_I5KlW2ag?e=vufGhW) |
| Matterport Pretrained 2D |  --  |  --  |  --  | [matterport_dps_120k.pth](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/EZtb_q9PgWhIl1Q6nSuV0bYBIVJUWnzMXh79002ZjEpwJA?e=hJMBVv) |
| Matterport Single-scale  | 16.58 | 44.26 | 36.68 | [matterport_full_single_scale.pth](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/ESZ7udkc4RxCkQcyLcZ3-08BjDDxkRHo6LRRoFqRnR_A3g?e=P88gGp) |

### Run

If you are using docker, you may set the following prefix for convenience.
```bash
export DOCKER_PREFIX="docker run -it --gpus all --shm-size 128G -v "$(pwd)":/workspace zx1239856/uni-3d:0.1.0"
```

#### Training 2D (Panoptic Segmentation/Depth) Model
```bash
$DOCKER_PREFIX OMP_NUM_THREADS=16 torchrun --nproc_per_node=8 train_net.py --config-file configs/front3d/mask2former_R50_bs16_160k.yaml OUTPUT_DIR <path-to-output-dir>
```

#### Training 3D Reconstruction Model
```bash
$DOCKER_PREFIX OMP_NUM_THREADS=16 torchrun --nproc_per_node=8 train_net.py --config-file configs/front3d/uni_3d_R50.yaml MODEL.WEIGHTS <path-to-pretrained-2d-model> OUTPUT_DIR <path-to-output-dir>
```

Use [uni_3d_R50_ms.yaml](configs/front3d/uni_3d_R50_ms.yaml) for multi-scale feature reprojection.

Please adjust `--nproc_per_node`, `OMP_NUM_THREADS` and `SOLVER.IMS_PER_BATCH` based on your environment.

#### Evaluate

Please add `--eval-only` flag to the training scripts above for evaluation.


#### Demo
You can generate meshes for visualization for 3D-FRONT images via the following command.

```bash
python demo_front3d.py -i <path-to-3d-front-image> -o <path-to-output-dir> -m <path-to-pretrained-model>
```


## Citation

Please consider citing Uni-3D if you find the work helpful.

```BibTeX
@InProceedings{Zhang_2023_ICCV,
    author    = {Zhang, Xiang and Chen, Zeyuan and Wei, Fangyin and Tu, Zhuowen},
    title     = {Uni-3D: A Universal Model for Panoptic 3D Scene Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {9256-9266}
}
```

## License
This repository is released under the Apache License 2.0. License can be found in [LICENSE](LICENSE) file.

## Acknowledgement

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) for the framework.
- [panoptic-reconstruction](https://github.com/xheon/panoptic-reconstruction) for the pre-processed 3D-FRONT and Matterport dataset, and evaluation codes.
- [BUOL](https://github.com/chtsy/buol) for generated depth and room mask on Matterport dataset.
