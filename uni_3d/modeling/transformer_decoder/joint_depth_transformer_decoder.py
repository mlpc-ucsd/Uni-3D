import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedJointTransformerDecoder(nn.Module):
    _version = 2

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        depth_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.depth_transformer_self_attention_layers = nn.ModuleList()
        self.depth_transformer_cross_attention_layers = nn.ModuleList()
        self.depth_transformer_ffn_layers = nn.ModuleList()

        self.cross_transformer_attention_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.depth_transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.depth_transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.depth_transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.cross_transformer_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.depth_decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.depth_query_feat = nn.Embedding(num_queries, hidden_dim)
        self.depth_query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.depth_level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        self.depth_input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                self.depth_input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
                weight_init.c2_xavier_fill(self.depth_input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
                self.depth_input_proj.append(nn.Sequential())

        self.hidden_dim = hidden_dim

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.depth_embed = MLP(hidden_dim, hidden_dim, depth_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["depth_dim"] = cfg.MODEL.SEM_SEG_HEAD.DEPTH_DIM

        return ret

    def forward(self, x, mask_features, depth_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        depth_src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            depth_src.append(self.depth_input_proj[i](x[i]).flatten(2) + self.depth_level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
            depth_src[-1] = depth_src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        depth_query_embed = self.depth_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        depth_output = self.depth_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_depth = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, outputs_depth, attn_mask, \
        decoder_out, depth_decoder_out = self.forward_prediction_heads(
            output, mask_features, 
            depth_output, depth_features,
            attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_depth.append(outputs_depth)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            depth_output = self.depth_transformer_cross_attention_layers[i](
                depth_output, depth_src[level_index],
                memory_mask=attn_mask, # we use masked attention where masks are from mask decoder
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=depth_query_embed  # memory pos is shared
            )

            # intra-query attention
            # [2, q, b, c] --> [2, q*b, c]
            cross_out = self.cross_transformer_attention_layers[i](
                torch.stack((output, depth_output)).flatten(1, 2),
                tgt_mask=None, tgt_key_padding_mask=None,
                query_pos=torch.stack((query_embed, depth_query_embed)).flatten(1, 2)
            ).reshape(2, self.num_queries, bs, self.hidden_dim)
            output, depth_output = cross_out.unbind()

            # inter-query attention

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
            )

            depth_output = self.depth_transformer_self_attention_layers[i](
                depth_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=depth_query_embed,
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](output)

            depth_output = self.depth_transformer_ffn_layers[i](depth_output)

            outputs_class, outputs_mask, outputs_depth, attn_mask, \
            decoder_out, depth_decoder_out = self.forward_prediction_heads(
                output, mask_features, 
                depth_output, depth_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_depth.append(outputs_depth)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_depths': predictions_depth[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_depth,
            ),
            'mask_features': mask_features,
            'depth_features': depth_features,
            'enc_features': x,
            'segm_decoder_out': decoder_out,
            'depth_decoder_out': depth_decoder_out,
        }
        return out

    def forward_prediction_heads(self, output, mask_features, depth_output, depth_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        depth_decoder_out = self.depth_decoder_norm(depth_output).transpose(0, 1)
        depth_embed = self.depth_embed(depth_decoder_out)
        outputs_depth = torch.einsum("bqc,bchw->bqhw", depth_embed, depth_features)

        outputs_depth = outputs_depth.sigmoid()

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, outputs_depth, attn_mask, decoder_output, depth_decoder_out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_depths": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_depth[:-1])
            ]
        else:
            return [
                {"pred_masks": b, "pred_depths": c} 
                for b, c in zip(outputs_seg_masks[:-1], outputs_depth[:-1])
            ]
