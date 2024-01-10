import torch
from torch import nn
from torch.nn import functional as F
import MinkowskiEngine as Me

from detectron2.config import configurable
from detectron2.projects.point_rend.point_features import point_sample
from .frustum import generate_frustum, compute_camera2frustum_transform


class SparseProjection(nn.Module):
    @configurable
    def __init__(self, 
                 truncation,
                 sign_channel=True,
                 depth_min=0.4, 
                 depth_max=6.0,
                 voxel_size=0.03,
                 frustum_dims=256):
        super().__init__()

        self.truncation = truncation
        self.sign_channel = sign_channel
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.voxel_size = voxel_size
        self.register_buffer("frustum_dimensions", torch.tensor([frustum_dims, frustum_dims, frustum_dims]))

    @classmethod
    def from_config(cls, cfg):
        return {
            "truncation": cfg.MODEL.UNI_3D.FRUSTUM3D.TRUNCATION,
            "sign_channel": cfg.MODEL.UNI_3D.PROJECTION.SIGN_CHANNEL,
            "voxel_size": cfg.MODEL.UNI_3D.PROJECTION.VOXEL_SIZE,
            "depth_min": cfg.MODEL.UNI_3D.PROJECTION.DEPTH_MIN,
            "depth_max": cfg.MODEL.UNI_3D.PROJECTION.DEPTH_MAX,
            "frustum_dims": cfg.MODEL.UNI_3D.FRUSTUM3D.GRID_DIMENSIONS,
        }
    
    @property
    def device(self):
        return self.frustum_dimensions.device
    
    @staticmethod
    def to_sparse_tensor(features, coordinates, stride=1):
        ms_sparse_features = torch.cat(features, dim=0)
        batched_coordinates = Me.utils.batched_coordinates(coordinates, device=ms_sparse_features.device)
        batched_coordinates[:, 1:] *= stride
        tensor = Me.SparseTensor(features=ms_sparse_features,
                                 coordinates=batched_coordinates,
                                 tensor_stride=stride,
                                 quantization_mode=Me.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        return tensor
    
    @staticmethod
    def projection(frustum, voxel_size, frustum_dimensions, truncation, intrinsic_inverse, depth, image_size, feat_size, near_clip, far_clip):
        camera2frustum, padding_offsets = compute_camera2frustum_transform(frustum, voxel_size, frustum_dimensions=frustum_dimensions)
        
        depth = depth.clone()
        depth[depth < near_clip] = 0
        depth[depth > far_clip] = 0
        depth_pixels_xy = depth.nonzero(as_tuple=False)
        device = depth_pixels_xy.device

        if depth_pixels_xy.shape[0] == 0:
            depth_pixels_xy = torch.tensor([[depth.shape[0] // 2, depth.shape[1] // 2]], device=device)
        depth_pixels_z = depth[depth_pixels_xy[:, 0], depth_pixels_xy[:, 1]].reshape(-1).float()

        depth_pixels_xy = depth_pixels_xy.flip(-1).float()
        normalized_depth_pixels_xy = depth_pixels_xy / torch.tensor([depth.shape[-1], depth.shape[-2]], device=device)
        xv, yv = (normalized_depth_pixels_xy * torch.tensor(image_size, device=device) * depth_pixels_z[:, None]).unbind(-1)
        # Use separate size for feature maps due to size divisibility padding
        feat_sampling_grid = depth_pixels_xy / torch.tensor(feat_size, device=device)
        
        depth_pixels = torch.stack([xv, yv, depth_pixels_z, torch.ones_like(depth_pixels_z)])
        pointcloud = torch.mm(intrinsic_inverse, depth_pixels.float())
        grid_coordinates = torch.mm(camera2frustum, pointcloud).t()[:, :3].contiguous()

        # projective sdf encoding
        # repeat truncation, add / subtract z-offset
        num_repetition = int(truncation * 2) + 1
        grid_coordinates = grid_coordinates.unsqueeze(1).repeat(1, num_repetition, 1)
        voxel_offsets = torch.arange(-truncation, truncation + 1, 1.0, device=device).view(1, -1, 1)
        coordinates_z = grid_coordinates[:, :, 2].clone()
        grid_coordinates[:, :, 2] += voxel_offsets[:, :, 0]

        num_points = grid_coordinates.size(0)

        flatten_coordinates = grid_coordinates.view(num_points * num_repetition, 3)
        # pad to 256,256,256
        flatten_coordinates = flatten_coordinates + padding_offsets
        return num_repetition, normalized_depth_pixels_xy, feat_sampling_grid, flatten_coordinates, coordinates_z, voxel_offsets

    def forward(self, multi_scale_features, encoder_features, batched_inputs) -> Me.SparseTensor:
        sparse_ms_coordinates = [[] for _ in range(len(multi_scale_features))]
        sparse_ms_features = [[] for _ in range(len(multi_scale_features))]
        sparse_enc_features = []
        sparse_enc_coordinates = []

        # Process each sample in the batch individually
        for idx, inputs in enumerate(batched_inputs):
            # Get GT intrinsic matrix
            intrinsic = inputs["intrinsic"].to(self.device)
            image_size = inputs["image_size"]  # (width, height)
            padded_size = inputs["padded_size"]
            intrinsic_inverse = torch.inverse(intrinsic)
            
            frustum = generate_frustum(image_size, intrinsic_inverse, self.depth_min, self.depth_max)
            
            num_repetition, segm_sampling_grid, feat_sampling_grid, flatten_coordinates, coordinates_z, voxel_offsets = \
                self.projection(frustum, self.voxel_size,
                                self.frustum_dimensions, self.truncation,
                                intrinsic_inverse, inputs["depth"], image_size, (padded_size[0] // 2, padded_size[1] // 2),
                                self.depth_min, self.depth_max)

            df_values = coordinates_z - coordinates_z.int()
            df_values = df_values + voxel_offsets.squeeze(-1)
            df_values.unsqueeze_(-1)

            # encode sign and values in 2 different channels
            if self.sign_channel:
                sign = torch.sign(df_values)
                value = torch.abs(df_values)
                df_values = torch.cat([sign, value], dim=-1)

            # segm features
            sem_seg = inputs["sem_seg"]
            sampled_segm_features = point_sample(sem_seg[None], segm_sampling_grid[None], align_corners=False)[0]
            
            # encoder features
            sampled_enc_features = point_sample(
                encoder_features[[idx]], 
                feat_sampling_grid[None], 
                align_corners=False,
            )[0]
            sampled_enc_features = torch.cat([sampled_enc_features, sampled_segm_features], dim=0)
            sampled_enc_features = sampled_enc_features.permute(1, 0).unsqueeze(1).repeat(1, num_repetition, 1)
            sampled_enc_features = torch.cat([df_values, sampled_enc_features], dim=-1)

            flat_features = sampled_enc_features.flatten(0, -2)
            sparse_enc_coordinates.append(flatten_coordinates)
            sparse_enc_features.append(flat_features)

            # multi-scale features
            for lvl, feat in enumerate(multi_scale_features):
                # TODO: 1/8, 1/4, 1/2, 1 levels -- 32, 64, 128, 256 size
                # TODO: remove hard-coded scales
                ratio = feat.shape[-1] / encoder_features.shape[-1]
                level_depth = F.interpolate(inputs["depth"][None, None], scale_factor=ratio, mode="nearest").squeeze()
                num_repetition, segm_sampling_grid, feat_sampling_grid, flatten_coordinates, *__ = \
                    self.projection(
                        frustum, self.voxel_size / ratio, self.frustum_dimensions * ratio,
                        round(ratio * self.truncation), intrinsic_inverse, level_depth,
                        image_size, (feat.shape[-1] * 2, feat.shape[-2] * 2),
                        self.depth_min, self.depth_max,
                    )
                sampled_features = point_sample(
                    feat[[idx]], 
                    feat_sampling_grid[None], 
                    align_corners=False,
                )[0]
                sampled_features = sampled_features.permute(1, 0).unsqueeze(1).repeat(1, num_repetition, 1).flatten(0, -2)
                sparse_ms_features[lvl].append(sampled_features)
                # Resize feature volume
                sparse_ms_coordinates[lvl].append(flatten_coordinates.clone())

        # Batch
        sparse_enc_features = self.to_sparse_tensor(sparse_enc_features, sparse_enc_coordinates)
        strides = [2, 4, 8]
        sparse_ms_features = [
            self.to_sparse_tensor(feats, coords, stride=stride) 
            for feats, coords, stride in zip(sparse_ms_features, sparse_ms_coordinates, strides)
        ]

        return sparse_ms_features, sparse_enc_features
