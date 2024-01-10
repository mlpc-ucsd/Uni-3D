from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
import MinkowskiEngine as Me
from .resnet import BasicBlock3D, SparseBasicBlock3D


def sparse_cat_union(a: Me.SparseTensor, b: Me.SparseTensor):
    cm = a.coordinate_manager
    assert cm == b.coordinate_manager, "different coords_man"
    assert a.tensor_stride == b.tensor_stride, "different tensor_stride"

    zeros_cat_with_a = torch.zeros([a.F.shape[0], b.F.shape[1]], dtype=a.dtype).to(a.device)
    zeros_cat_with_b = torch.zeros([b.F.shape[0], a.F.shape[1]], dtype=a.dtype).to(a.device)

    feats_a = torch.cat([a.F, zeros_cat_with_a], dim=1)
    feats_b = torch.cat([zeros_cat_with_b, b.F], dim=1)

    new_a = Me.SparseTensor(
        features=feats_a,
        coordinates=a.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    new_b = Me.SparseTensor(
        features=feats_b,
        coordinates=b.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    return new_a + new_b


class SparseToDense(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        assert len(input_size) == 3
        self.input_size = input_size

    def forward(self, feature: Me.SparseTensor) -> torch.Tensor:
        batch_size = len(feature.decomposed_coordinates_and_features[0])
        feat_dim = feature.C.shape[-1]

        out_size = (torch.div(torch.tensor(self.input_size), torch.tensor(feature.tensor_stride), rounding_mode="floor")).tolist()
        shape = torch.Size([batch_size, feat_dim, *out_size])
        min_coordinate = torch.IntTensor([0, 0, 0])

        mask = (feature.C[:, 1] < self.input_size[0]) & \
               (feature.C[:, 2] < self.input_size[1]) & \
               (feature.C[:, 3] < self.input_size[2])
        mask = mask & (feature.C[:, 1] >= 0) & (feature.C[:, 2] >= 0) & (feature.C[:, 3] >= 0)

        feature = Me.MinkowskiPruning()(feature, mask)

        dense = feature.dense(shape, min_coordinate=min_coordinate)[0]
        
        return dense


class FrustumDecoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        num_output_features = cfg.MODEL.UNI_3D.FRUSTUM3D.UNET_OUTPUT_CHANNELS
        num_features=cfg.MODEL.UNI_3D.FRUSTUM3D.UNET_FEATURES
        sign_channel = cfg.MODEL.UNI_3D.PROJECTION.SIGN_CHANNEL
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        depth_dim = cfg.MODEL.SEM_SEG_HEAD.DEPTH_DIM
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        frustum_dims = cfg.MODEL.UNI_3D.FRUSTUM3D.GRID_DIMENSIONS
        frustum_dims = [frustum_dims] * 3
        self.use_ms_features = cfg.MODEL.UNI_3D.FRUSTUM3D.USE_MULTI_SCALE
        self.truncation = cfg.MODEL.UNI_3D.FRUSTUM3D.TRUNCATION

        ms_feature_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        # Input encoding
        self.input_dims = [2 if sign_channel else 1, mask_dim + depth_dim, num_classes] 
        self.input_encoders = nn.ModuleList()
        for input_dim in self.input_dims:
            downsample = nn.Sequential(
                Me.MinkowskiConvolution(input_dim, num_features, kernel_size=1, stride=1, bias=True, dimension=3),
                Me.MinkowskiInstanceNorm(num_features),
            )
            self.input_encoders.append(SparseBasicBlock3D(input_dim, num_features, downsample=downsample))
        
        self.level_encoders = nn.ModuleList([
            self.make_encoder(len(self.input_encoders) * num_features, num_features),  # 256 --> 128
            self.make_encoder(num_features, num_features*2),                           # 128 --> 64
            self.make_encoder(num_features*2, num_features*4, is_sparse=False),        # 64 --> 32
            self.make_encoder(num_features*4, num_features*8, is_sparse=False),
            self.make_encoder(num_features*8, num_features*8, is_sparse=False),
        ])

        sparse_to_dense = SparseToDense(frustum_dims)

        if self.use_ms_features:
            self.feature_adapters = nn.ModuleList([
                self.make_adapter(ms_feature_channels, num_features),                      # level 128
                self.make_adapter(ms_feature_channels, num_features*2),                    # level 64
                self.make_adapter(ms_feature_channels, num_features*4, [sparse_to_dense]), # level 32
            ])
        else:
            self.feature_adapters = None

        self.enc_level_conversion = nn.ModuleList([
            nn.Identity(),
            sparse_to_dense,
            nn.Identity(),
            nn.Identity(),
        ])

        self.level_decoders = nn.ModuleList([
            self.make_decoder(num_features*3, num_output_features),
            self.make_decoder(num_features*6, num_features*2, 
                              extra_layers=[SparseBasicBlock3D(num_features*2, num_features*2)]),
            self.make_decoder(num_features*8, num_features*2, is_sparse=False),
            self.make_decoder(num_features*16, num_features*4, is_sparse=False),
            self.make_decoder(num_features*8, num_features*8, is_sparse=False),
        ])

        # occupancy heads
        self.level_occupancy_heads = nn.ModuleList([
            nn.Sequential(
                Me.MinkowskiInstanceNorm(num_output_features),
                Me.MinkowskiReLU(inplace=True),
                SparseBasicBlock3D(num_output_features, num_output_features),
                Me.MinkowskiConvolution(num_output_features, 1, kernel_size=3, bias=True, dimension=3),
            ),
            Me.MinkowskiLinear(num_features*2, 1),
            nn.Linear(num_features*4, 1),
        ])

        # panoptic heads
        self.level_segm_embeddings = nn.ModuleList([
            nn.Sequential(
                Me.MinkowskiInstanceNorm(num_output_features),
                Me.MinkowskiReLU(inplace=True),
                SparseBasicBlock3D(num_output_features, num_output_features),
            ),
            SparseBasicBlock3D(num_features*3, num_features*3),
            nn.Sequential(
                BasicBlock3D(num_features*4, num_features*4),
                BasicBlock3D(num_features*4, num_features*4),
            )
        ])
        self.level_segm_query_projection = nn.ModuleList([
            nn.Linear(mask_dim, num_output_features),
            nn.Linear(mask_dim, num_features*3),
            nn.Linear(mask_dim, num_features*4),
        ])

        # geometry head
        self.geometry_head = nn.Sequential(
            Me.MinkowskiInstanceNorm(num_output_features),
            Me.MinkowskiReLU(inplace=True),
            SparseBasicBlock3D(num_output_features, num_output_features),
            Me.MinkowskiConvolution(num_output_features, 1, kernel_size=3, bias=True, dimension=3),
        )

        self.register_buffer("frustum_dimensions", torch.tensor(frustum_dims))

    @staticmethod
    def forward_sparse_segm(segm_features, queries):
        features = segm_features.decomposed_features
        segms = torch.cat([torch.mm(features[idx], queries[idx].T) for idx in range(len(features))], dim=0)
        return Me.SparseTensor(segms,
                               coordinate_manager=segm_features.coordinate_manager, 
                               coordinate_map_key=segm_features.coordinate_map_key)

    @staticmethod
    def make_encoder(input_dim, output_dim, is_sparse=True):
        if is_sparse:
            downsample = nn.Sequential(
                Me.MinkowskiConvolution(input_dim, output_dim, kernel_size=4, stride=2, bias=True, dimension=3),
                Me.MinkowskiInstanceNorm(output_dim),
            )
            module = nn.Sequential(
                SparseBasicBlock3D(input_dim, output_dim, stride=2, downsample=downsample),
                SparseBasicBlock3D(output_dim, output_dim),
            )
        else:
            downsample = nn.Conv3d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False)
            module = nn.Sequential(
                BasicBlock3D(input_dim, output_dim, stride=2, downsample=downsample),
                BasicBlock3D(output_dim, output_dim),
            )
        return module
    
    @staticmethod
    def make_decoder(input_dim, output_dim, is_sparse=True, extra_layers: Optional[List]=None):
        if extra_layers is None:
            extra_layers = []
        if is_sparse:
            return nn.Sequential(
                Me.MinkowskiConvolutionTranspose(input_dim, output_dim, kernel_size=4, stride=2, bias=False, dimension=3, expand_coordinates=True),
                Me.MinkowskiInstanceNorm(output_dim),
                Me.MinkowskiReLU(inplace=True),
                *extra_layers,
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose3d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(output_dim),
                nn.ReLU(inplace=True),
                *extra_layers,
            )

    @staticmethod
    def make_adapter(input_dim, output_dim, extra_layers: Optional[List]=None):
        if extra_layers is None:
            extra_layers = []
        downsample = nn.Sequential(
            Me.MinkowskiConvolution(input_dim, output_dim, kernel_size=1, stride=1, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(output_dim),
        )
        return nn.Sequential(
            SparseBasicBlock3D(input_dim, output_dim, downsample=downsample),
            *extra_layers,
        )

    def forward(self, ms_features: List[Me.SparseTensor], features: Me.SparseTensor, segm_queries, frustum_mask):
        start_dim = 0
        encoded_inputs = []
        cm = features.coordinate_manager
        key = features.coordinate_map_key
        for dim, encoder in zip(self.input_dims, self.input_encoders):
            encoded_inputs.append(
                encoder(
                    Me.SparseTensor(features.F[:, start_dim:start_dim + dim], coordinate_manager=cm, coordinate_map_key=key)
                )
            )
            start_dim += dim
        encoded_inputs = Me.cat(*encoded_inputs)

        lvls = len(self.level_encoders)

        # high to low resolution
        encoder_outputs = []
        encoder_inputs = [encoded_inputs]
        
        for idx in range(len(self.level_encoders)):
            encoded = self.level_encoders[idx](encoder_inputs[idx])
            if self.use_ms_features and idx < len(self.feature_adapters):
                feat = self.feature_adapters[idx](ms_features[idx])
                if isinstance(encoded, torch.Tensor):
                    encoded = encoded + feat
                else:
                    feat = Me.SparseTensor(feat.F, 
                                           coordinates=feat.C,
                                           tensor_stride=feat.tensor_stride,
                                           coordinate_manager=encoded.coordinate_manager)
                    encoded = encoded + feat
            encoder_outputs.append(encoded)
            if idx < lvls - 1:
                encoder_inputs.append(self.enc_level_conversion[idx](encoded))

        # low to high resolution
        decoder_outputs = []
        decoder_inputs = [encoder_outputs[-1]]
        
        pred_occupancies = []
        pred_segms = []
        pred_geometry = None
        
        # U-Net
        for idx in reversed(range(lvls)):
            decoded = self.level_decoders[idx](decoder_inputs[lvls - 1 - idx])
            decoder_outputs.append(decoded)
            
            if idx <= 1:
                # level 128, 256
                occupancy = self.level_occupancy_heads[idx](decoded)
                # mask invalid voxels outside of frustum
                valid_mask = ((occupancy.C[:, 1:] >= 0) & (occupancy.C[:, 1:] < self.frustum_dimensions)).all(-1)
                pred_occupancies.append(Me.MinkowskiPruning()(occupancy, valid_mask))
                pruning_mask = (Me.MinkowskiSigmoid()(occupancy).F.squeeze(-1) > 0.5) & valid_mask
                sparse_out = Me.MinkowskiPruning()(decoded, pruning_mask)
                
                if idx > 0:
                    # level 128
                    sparse_out = sparse_cat_union(encoder_outputs[idx-1], sparse_out)
                    valid_mask = ((sparse_out.C[:, 1:] >= 0) & (sparse_out.C[:, 1:] < self.frustum_dimensions)).all(-1)
                    decoder_inputs.append(Me.MinkowskiPruning()(sparse_out, valid_mask))
                else:
                    # level 256
                    pred_geometry = self.geometry_head(sparse_out)
                    predicted_values = pred_geometry.F
                    predicted_values = torch.clamp(predicted_values, 0.0, self.truncation)
                    pred_geometry = Me.SparseTensor(
                        predicted_values,
                        coordinate_manager=pred_geometry.coordinate_manager,
                        coordinate_map_key=pred_geometry.coordinate_map_key,
                    )
                    valid_mask = ((pred_geometry.C[:, 1:] >= 0) & (pred_geometry.C[:, 1:] < self.frustum_dimensions)).all(-1)
                    pred_geometry = Me.MinkowskiPruning()(pred_geometry, valid_mask)
                
                queries = self.level_segm_query_projection[idx](segm_queries)
                segm_features = self.level_segm_embeddings[idx](sparse_out)
                pred_segm = self.forward_sparse_segm(segm_features, queries)
                valid_mask = ((pred_segm.C[:, 1:] >= 0) & (pred_segm.C[:, 1:] < self.frustum_dimensions)).all(-1)
                pred_segms.append(Me.MinkowskiPruning()(pred_segm, valid_mask))
            
            elif idx == 2:
                # level 64
                decoded = torch.cat([encoder_inputs[idx], decoded], dim=1)
                occupancy = self.level_occupancy_heads[idx](decoded.permute(0, 2, 3, 4, 1)).squeeze(-1)
                pred_occupancies.append(occupancy.masked_fill(~frustum_mask.squeeze(1), -torch.inf))

                queries = self.level_segm_query_projection[idx](segm_queries)
                segm_features = self.level_segm_embeddings[idx](decoded)
                pred_segm = torch.einsum("bqc,bchwd->bqhwd", queries, segm_features)
                pred_segms.append(pred_segm.masked_fill(~frustum_mask, -torch.inf))

                pruning_mask = (occupancy.sigmoid() > 0.5) & frustum_mask.squeeze(1)
                coords = pruning_mask.nonzero()
                sparse_out = decoded[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]]
                encoded = encoder_outputs[idx-1]
                stride = encoded.tensor_stride
                coords = coords.clone()
                coords[:, 1:] *= torch.tensor(stride, device=coords.device)
                sparse_out = Me.SparseTensor(sparse_out, coordinates=coords.int().contiguous(), tensor_stride=stride, coordinate_manager=cm)
                decoder_inputs.append(sparse_cat_union(encoded, sparse_out))
            else:
                decoder_inputs.append(torch.cat([encoder_inputs[idx], decoded], dim=1))

        return {
            "pred_geometry": pred_geometry,
            "pred_occupancies": pred_occupancies,
            "pred_segms": pred_segms,
        }
