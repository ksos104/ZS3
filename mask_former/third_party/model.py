from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# TODO: this attentionpool2d may need re written and re training
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, masks=None):
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        # TODO: resize positional embedding and add the positional embedding
        if masks == None:
            # x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
            x = self.v_proj(x)
            x = self.c_proj(x)
            out_c = self.c_proj.weight.shape[0]
            # x = x[:-1, :, :]
            # x = x[1:, :, :]
            x = x.reshape(H, W, N, out_c)
            return x
        else:
            masks = masks.sigmoid()
            # TODO: try different structures
            pipeline = "way4"
            assert pipeline in ["way1", "way2", "way3", "way4"]
            if pipeline == "way1":
                num_queries = masks.shape[0]
                # [num_queries, H, W] -> [H*W, num_queries, 1]
                masks = masks.reshape(masks.shape[0], masks.shape[1] * masks.shape[2]).permute(1, 0).unsqueeze(2)
                # [H*W, 1, C]
                assert x.shape[1] == 1
                # import ipdb; ipdb.set_trace()
                masks_weights = torch.sum(masks, dim=0)
                # TODO: check "+0.001"
                mask_mean = torch.sum(masks * x, dim=0) / (masks_weights + 0.001)
                mask_mean = mask_mean.unsqueeze(1)
                # TODO: change the mask_mean[0] to mask_mean
                # x = torch.cat([mask_mean[0].unsqueeze(0), x], dim=0) # (HW+1)NC
                x = torch.cat([mask_mean, x], dim=0) # (HW+num_queries)NC
                # import ipdb; ipdb.set_trace()
                # TODO: change attention mask 1. maskmean和maskmean之间的attention为-inf
                #  2. change the attention_mask according to masks
                attn_mask = torch.zeros(x.shape[0], x.shape[0]).to(mask_mean.device)
                attn_mask[:num_queries, : num_queries] = 1
                attn_mask = attn_mask.bool()
                x, _ = F.multi_head_attention_forward(
                    query=x[:num_queries], key=x, value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    in_proj_weight=None,
                    in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight,
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                    attn_mask=attn_mask
                )
                return x[:num_queries]
            elif pipeline == "way2":
                num_queries = masks.shape[0]
                # [num_queries, H, W] -> [H*W, num_queries, 1]
                masks = masks.reshape(masks.shape[0], masks.shape[1] * masks.shape[2]).permute(1, 0).unsqueeze(2)
                # [H*W, 1, C]
                assert x.shape[1] == 1
                # import ipdb; ipdb.set_trace()
                masks_weights = torch.sum(masks, dim=0)
                # TODO: check "+0.001"
                mask_mean = torch.sum(masks * x, dim=0) / (masks_weights + 0.001)
                mask_mean = mask_mean.unsqueeze(1)
                # TODO: change the mask_mean[0] to mask_mean
                # x = torch.cat([mask_mean[0].unsqueeze(0), x], dim=0) # (HW+1)NC
                x = torch.cat([mask_mean, x], dim=0)  # (HW+num_queries)NC
                # import ipdb; ipdb.set_trace()
                # TODO: change attention mask 1. maskmean和maskmean之间的attention为-inf
                #  2. change the attention_mask according to masks
                attn_mask = torch.zeros(num_queries, x.shape[0]).to(mask_mean.device)
                attn_mask[:num_queries, :num_queries] = 1
                attn_mask[:num_queries, :num_queries] = attn_mask[:num_queries, :num_queries] - torch.eye(num_queries).to(mask_mean.device)
                attn_mask = attn_mask.bool()
                # import ipdb; ipdb.set_trace()
                x, _ = F.multi_head_attention_forward(
                    query=x[:num_queries], key=x, value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    in_proj_weight=None,
                    in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight,
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                    attn_mask=attn_mask
                )
                return x[:num_queries]
            elif pipeline == "way3":
                num_queries = masks.shape[0]
                # [num_queries, H, W] -> [H*W, num_queries, 1]
                masks = masks.reshape(masks.shape[0], masks.shape[1] * masks.shape[2]).permute(1, 0).unsqueeze(2)
                # [H*W, 1, C]
                assert x.shape[1] == 1
                # import ipdb; ipdb.set_trace()
                # way3 has a bug, TODO: the fix the bug in the following line
                masks_weights = torch.sum(masks, dim=0)
                # TODO: check "+0.001"
                mask_mean = torch.sum(masks * x, dim=0) / (masks_weights + 0.001)
                mask_mean = mask_mean.unsqueeze(1)
                # TODO: change the mask_mean[0] to mask_mean
                # x = torch.cat([mask_mean[0].unsqueeze(0), x], dim=0) # (HW+1)NC
                x = torch.cat([mask_mean, x], dim=0)  # (HW+num_queries)NC
                # import ipdb; ipdb.set_trace()
                # TODO: change attention mask 1. maskmean和maskmean之间的attention为-inf
                #  2. change the attention_mask according to masks
                attn_mask = torch.zeros(num_queries, x.shape[0]).to(mask_mean.device)
                attn_mask[:num_queries, :num_queries] = 1
                attn_mask[:num_queries, :num_queries] = attn_mask[:num_queries, :num_queries] - torch.eye(num_queries).to(mask_mean.device)
                # import ipdb; ipdb.set_trace()
                attn_mask[:, num_queries:] = masks.squeeze().permute(1, 0) < 0.9
                attn_mask = attn_mask.bool()
                # import ipdb; ipdb.set_trace()
                x, _ = F.multi_head_attention_forward(
                    query=x[:num_queries], key=x, value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    in_proj_weight=None,
                    in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight,
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                    attn_mask=attn_mask
                )
                return x[:num_queries]
            elif pipeline == "way4":
                num_queries = masks.shape[0]
                # [num_queries, H, W] -> [H*W, num_queries, 1]
                masks = masks.reshape(masks.shape[0], masks.shape[1] * masks.shape[2]).permute(1, 0).unsqueeze(2)
                # [H*W, 1, C]
                assert x.shape[1] == 1
                # import ipdb; ipdb.set_trace()
                masks_weights = torch.sum(masks, dim=0)
                # TODO: check "+0.001"
                mask_mean = torch.sum(masks * x, dim=0) / (masks_weights + 0.001)
                mask_mean = mask_mean.unsqueeze(1)
                # TODO: change the mask_mean[0] to mask_mean
                # x = torch.cat([mask_mean[0].unsqueeze(0), x], dim=0) # (HW+1)NC
                x = torch.cat([mask_mean, x], dim=0)  # (HW+num_queries)NC
                # import ipdb; ipdb.set_trace()
                # TODO: change attention mask 1. maskmean和maskmean之间的attention为-inf
                #  2. change the attention_mask according to masks
                attn_mask = torch.zeros(num_queries, x.shape[0]).to(mask_mean.device)
                attn_mask[:num_queries, :num_queries] = 1
                attn_mask[:num_queries, :num_queries] = attn_mask[:num_queries, :num_queries] - torch.eye(num_queries).to(mask_mean.device)
                # import ipdb; ipdb.set_trace()
                attn_mask[:, num_queries:] = masks.squeeze().permute(1, 0) <= 0.9
                attn_mask = attn_mask.bool()
                # import ipdb; ipdb.set_trace()
                x, _ = F.multi_head_attention_forward(
                    query=x[:num_queries], key=x, value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    in_proj_weight=None,
                    in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight,
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                    attn_mask=attn_mask
                )
                return x[:num_queries]
            # x, _ = F.multi_head_attention_forward(
            #     query=x, key=x, value=x,
            #     embed_dim_to_check=x.shape[-1],
            #     num_heads=self.num_heads,
            #     q_proj_weight=self.q_proj.weight,
            #     k_proj_weight=self.k_proj.weight,
            #     v_proj_weight=self.v_proj.weight,
            #     in_proj_weight=None,
            #     in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            #     bias_k=None,
            #     bias_v=None,
            #     add_zero_attn=False,
            #     dropout_p=0,
            #     out_proj_weight=self.c_proj.weight,
            #     out_proj_bias=self.c_proj.bias,
            #     use_separate_proj_weight=True,
            #     training=self.training,
            #     need_weights=False
            # )


# Note: this is reimplementation of DenseCLIP:
# a modified version of ResNet CLIP
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

        self.embed_dim = embed_dim

    def reset_resolution(self, resolution):
        self.input_resolution = resolution
        self.attnpool = AttentionPool2d(self.input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, masks=None):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # shape of x: [bs, channel, feat_h, feat_w], e.g., [100, 3072, 7, 7]
        # import ipdb; ipdb.set_trace()
        x= self.attnpool(x, masks)
        # import ipdb; ipdb.set_trace()
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, masks=None):
        if masks == None:
            return self.visual(image.type(self.dtype))
        else:
            return self.visual(image.type(self.dtype), masks.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # import pdb; pdb.set_trace()
        # normalized features
        # image_features shape: [1, 1024]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_iamge = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_iamge, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
