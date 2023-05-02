# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.position_encoding import PositionEmbeddingSine
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer

import torch


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model

class DINODecoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        '''
            Modified for DINO decoder
        '''
        super().__init__()
        
        # self.in_features = ['features']
        # feature_channels = [384]
        self.n_layers = 4
        # self.n_heads = 6
        # self.n_heads_list = [6, 6, 6, 1, 1]
        self.n_heads_list = [6, 6, 6, 6, 6]
        self.conv_dim = 384

        attn_convs = []
        feature_convs = []

        use_bias = norm == ""
        for idx in range(self.n_layers):
            attn_norm = get_norm("LN", self.n_heads_list[idx+1])
            feature_norm = get_norm(norm, self.conv_dim)

            attn_conv = Conv2d(
                self.n_heads_list[idx], self.n_heads_list[idx+1], kernel_size=1, bias=use_bias, norm=attn_norm
            )
            feature_conv = Conv2d(
                self.conv_dim+self.n_heads_list[idx],
                self.conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=feature_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(attn_conv)
            weight_init.c2_xavier_fill(feature_conv)
            self.add_module("adapter_{}".format(idx + 1), attn_conv)
            self.add_module("layer_{}".format(idx + 1), feature_conv)

            attn_convs.append(attn_conv)
            feature_convs.append(feature_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.attn_convs = attn_convs
        self.feature_convs = feature_convs

        # self.mask_dim = mask_dim
        # self.mask_features = Conv2d(
        #     conv_dim,
        #     mask_dim,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        # )
        # weight_init.c2_xavier_fill(self.mask_features)
        

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features, attentions):
        # Reverse feature maps into top-down order (from low to high resolution)
        bs = features.shape[0]
        features = features.permute(0,2,1).reshape(bs, self.conv_dim, 60, 60)
        attentions = attentions.reshape(bs, self.n_heads_list[0], 60, 60)
        
        for idx in range(self.n_layers):
            features = torch.cat((features, attentions), dim=1)
            
            features = self.feature_convs[idx](features)
            # attentions = self.attn_convs[idx](attentions)
            
            # attentions = attentions.reshape(bs, self.n_heads_list[idx+1], -1)
            # attentions = attentions.softmax(dim=-1)
            # attentions = attentions.reshape(bs, self.n_heads_list[idx+1], 60, 60)
            
        features = features.reshape(bs, self.conv_dim, 3600).permute(0,2,1)
        attentions = attentions.reshape(bs, self.n_heads_list[-1], 3600)
        # attentions = attentions.softmax(dim=-1)
        
        return features, attentions

    def forward(self, features, attentions, targets=None):
        return self.forward_features(features, attentions)


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)

# class TransformerEncoderPixelDecoder(BasePixelDecoder):
#     @configurable
#     def __init__(
#         self,
#         input_shape: Dict[str, ShapeSpec],
#         *,
#         transformer_dropout: float,
#         transformer_nheads: int,
#         transformer_dim_feedforward: int,
#         transformer_enc_layers: int,
#         transformer_pre_norm: bool,
#         conv_dim: int,
#         mask_dim: int,
#         norm: Optional[Union[str, Callable]] = None,
#     ):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             input_shape: shapes (channels and stride) of the input features
#             transformer_dropout: dropout probability in transformer
#             transformer_nheads: number of heads in transformer
#             transformer_dim_feedforward: dimension of feedforward network
#             transformer_enc_layers: number of transformer encoder layers
#             transformer_pre_norm: whether to use pre-layernorm or not
#             conv_dims: number of output channels for the intermediate conv layers.
#             mask_dim: number of output channels for the final conv layer.
#             norm (str or callable): normalization for all conv layers
#         """
#         super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

#         input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
#         self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
#         feature_strides = [v.stride for k, v in input_shape]
#         feature_channels = [v.channels for k, v in input_shape]

#         in_channels = feature_channels[len(self.in_features) - 1]
#         self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
#         weight_init.c2_xavier_fill(self.input_proj)
#         self.transformer = TransformerEncoderOnly(
#             d_model=conv_dim,
#             dropout=transformer_dropout,
#             nhead=transformer_nheads,
#             dim_feedforward=transformer_dim_feedforward,
#             num_encoder_layers=transformer_enc_layers,
#             normalize_before=transformer_pre_norm,
#         )
#         N_steps = conv_dim // 2
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

#         # update layer
#         use_bias = norm == ""
#         output_norm = get_norm(norm, conv_dim)
#         output_conv = Conv2d(
#             conv_dim,
#             conv_dim,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=use_bias,
#             norm=output_norm,
#             activation=F.relu,
#         )
#         weight_init.c2_xavier_fill(output_conv)
#         delattr(self, "layer_{}".format(len(self.in_features)))
#         self.add_module("layer_{}".format(len(self.in_features)), output_conv)
#         self.output_convs[0] = output_conv

#     @classmethod
#     def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
#         ret = super().from_config(cfg, input_shape)
#         ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
#         ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
#         ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
#         ret[
#             "transformer_enc_layers"
#         ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
#         ret["transformer_pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
#         return ret

#     def forward_features(self, features):
#         # Reverse feature maps into top-down order (from low to high resolution)
#         for idx, f in enumerate(self.in_features[::-1]):
#             x = features[f]
#             lateral_conv = self.lateral_convs[idx]
#             output_conv = self.output_convs[idx]
#             if lateral_conv is None:
#                 transformer = self.input_proj(x)
#                 pos = self.pe_layer(x)
#                 transformer = self.transformer(transformer, None, pos)
#                 y = output_conv(transformer)
#                 # save intermediate feature as input to Transformer decoder
#                 transformer_encoder_features = transformer
#             else:
#                 cur_fpn = lateral_conv(x)
#                 # Following FPN implementation, we use nearest upsampling here
#                 y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
#                 y = output_conv(y)
#         # TODO: modify this and output multi level transformer encoder features.
#         return self.mask_features(y), transformer_encoder_features

#     def forward(self, features, targets=None):
#         logger = logging.getLogger(__name__)
#         logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
#         return self.forward_features(features)
