import math
import os
import sys
from functools import lru_cache
from collections.abc import Callable

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "DINOv3_vits16": "",
    "DINOv3_vits14": "",
    "DINOv3_vitb16": "",
    "DINOv3_vitb14": "",
    "DINOv3_vitg14": "",
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    # Original PyTorch: def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor
    if drop_prob == 0. or not training:  # if drop_prob == 0.0 or not training: return input
        return x
    keep_prob = paddle.full(shape=[], fill_value=1 - drop_prob, dtype=x.dtype)  # keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)  # random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor = paddle.floor(random_tensor)  # random_tensor.floor_()
    output = x.divide(keep_prob) * random_tensor  # output = input.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def rotate_half(x):
    # Original PyTorch: def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]  # x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]  # x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # return torch.cat((-x2, x1), dim=-1)


class LayerScale(nn.Layer):
    # Original PyTorch: class DINOv3ViTLayerScale(nn.Module)
    def __init__(self, config):
        super().__init__()
        # self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))
        self.lambda1 = self.create_parameter(
            shape=[config.hidden_size],
            default_initializer=Constant(value=config.layerscale_value))

    def forward(self, hidden_state):
        return hidden_state * self.lambda1  # return hidden_state * self.lambda1


class DINOv3ViTEmbeddings(nn.Layer):
    # Original PyTorch: class DINOv3ViTEmbeddings(nn.Module)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        # Original: self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # Note: PyTorch uses randn with std=1.0, PaddlePaddle uses Normal(std=1.0)
        self.cls_token = self.create_parameter(
            shape=[1, 1, config.hidden_size],
            default_initializer=Normal(std=1.0))
        self.add_parameter("cls_token", self.cls_token)

        # Original: self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = self.create_parameter(
            shape=[1, 1, config.hidden_size],
            default_initializer=zeros_)
        self.add_parameter("mask_token", self.mask_token)

        # Original: self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))
        # Note: PyTorch initializes with trunc_normal_ in _init_weights, PaddlePaddle uses Normal(std=0.02) here
        if config.num_register_tokens > 0:
            self.register_tokens = self.create_parameter(
                shape=[1, config.num_register_tokens, config.hidden_size],
                default_initializer=Normal(std=0.02))
            self.add_parameter("register_tokens", self.register_tokens)
        else:
            self.register_tokens = paddle.zeros([1, 0, config.hidden_size])

        # Original: self.patch_embeddings = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.patch_embeddings = nn.Conv2D(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size)

    def forward(self, pixel_values, bool_masked_pos=None):
        # Original PyTorch: def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor
        batch_size = pixel_values.shape[0]

        # Original: patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        # Original: patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        patch_embeddings = self.patch_embeddings(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2).transpose([0, 2, 1])  # flatten(2) then transpose [0,2,1]

        # Original: patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)
        if bool_masked_pos is not None:
            mask_token = self.mask_token.astype(patch_embeddings.dtype)  # mask_token.to(patch_embeddings.dtype)
            patch_embeddings = paddle.where(
                bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)

        # Original: cls_token = self.cls_token.expand(batch_size, -1, -1)
        # Original: register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        # Original: embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)
        cls_token = self.cls_token.expand([batch_size, -1, -1]).astype(patch_embeddings.dtype)
        register_tokens = self.register_tokens.expand([batch_size, -1, -1]).astype(patch_embeddings.dtype)
        embeddings = paddle.concat([cls_token, register_tokens, patch_embeddings], axis=1)

        return embeddings


@lru_cache(maxsize=32)
def get_patches_center_coordinates(num_patches_h, num_patches_w, dtype_str):
    # Original PyTorch: @compile_compatible_method_lru_cache(maxsize=32)
    # Original: def get_patches_center_coordinates(num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor
    
    # Original: coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    # Original: coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = paddle.arange(0.5, num_patches_h, dtype='float32')
    coords_w = paddle.arange(0.5, num_patches_w, dtype='float32')
    
    # Original: coords_h = coords_h / num_patches_h
    # Original: coords_w = coords_w / num_patches_w
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    
    # Original: coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    # Original: coords = coords.flatten(0, 1)
    mesh_h, mesh_w = paddle.meshgrid(coords_h, coords_w)
    coords = paddle.stack([mesh_h, mesh_w], axis=-1)
    coords = coords.reshape([-1, 2])  # flatten(0, 1)
    
    # Original: coords = 2.0 * coords - 1.0
    coords = 2.0 * coords - 1.0
    return coords


def augment_patches_center_coordinates(coords, shift=None, jitter=None, rescale=None):
    # Original PyTorch: def augment_patches_center_coordinates(coords, shift, jitter, rescale)
    
    # Original: shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
    # Original: shift_hw = shift_hw.uniform_(-shift, shift)
    if shift is not None:
        shift_hw = paddle.uniform([1, 2], min=-shift, max=shift, dtype=coords.dtype)
        coords = coords + shift_hw  # coords = coords + shift_hw

    # Original: jitter_range = np.log(jitter)
    # Original: jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
    # Original: jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
    if jitter is not None:
        jitter_range = np.log(jitter)
        jitter_hw = paddle.uniform([1, 2], min=-jitter_range, max=jitter_range, dtype=coords.dtype).exp()
        coords = coords * jitter_hw  # coords = coords * jitter_hw

    # Original: rescale_range = np.log(rescale)
    # Original: rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype)
    # Original: rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
    if rescale is not None:
        rescale_range = np.log(rescale)
        rescale_hw = paddle.uniform([1], min=-rescale_range, max=rescale_range, dtype=coords.dtype).exp()
        coords = coords * rescale_hw  # coords = coords * rescale_hw

    return coords


class DINOv3ViTRopePositionEmbedding(nn.Layer):
    # Original PyTorch: class DINOv3ViTRopePositionEmbedding(nn.Module)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = config.rope_theta  # self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Original: inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        inv_freq = 1 / (self.base ** paddle.arange(0, 1, 4 / self.head_dim, dtype='float32'))
        self.register_buffer("inv_freq", inv_freq, persistable=False)  # self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values):
        # Original: def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_size
        num_patches_w = width // self.config.patch_size

        dtype_str = str(pixel_values.dtype)
        # Original: patch_coords = get_patches_center_coordinates(num_patches_h, num_patches_w, dtype=torch.float32, device=device)
        coords = get_patches_center_coordinates(num_patches_h, num_patches_w, dtype_str)
        
        # Original: if self.training: patch_coords = augment_patches_center_coordinates(...)
        if self.training:
            shift = getattr(self.config, 'pos_embed_shift', None)
            jitter = getattr(self.config, 'pos_embed_jitter', None)
            rescale = getattr(self.config, 'pos_embed_rescale', None)
            coords = augment_patches_center_coordinates(coords, shift, jitter, rescale)

        # Original: angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        # Critical fix: angles.flatten(1, 2) then tile instead of reshape
        # Original: angles = angles.flatten(1, 2)
        # Original: angles = angles.tile(2)
        angles = angles.flatten(1, 2)  # (height * width, 2, head_dim/4) -> (height * width, head_dim/2)
        angles = paddle.tile(angles, [1, 2])  # (height * width, head_dim/2) -> (height * width, head_dim)

        # Original: cos = torch.cos(angles); sin = torch.sin(angles)
        cos = paddle.cos(angles)
        sin = paddle.sin(angles)

        dtype = pixel_values.dtype
        return cos.astype(dtype), sin.astype(dtype)  # return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Original PyTorch: def apply_rotary_pos_emb(q, k, cos, sin)
    # Applies RoPE only to patch tokens, ignoring prefix tokens (cls + register tokens)
    
    num_tokens = q.shape[-2]  # num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]  # num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches  # cls token + register tokens

    # Original: q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    # Original: k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)
    q_prefix_tokens, q_patches = q.split([num_prefix_tokens, num_patches], axis=-2)
    k_prefix_tokens, k_patches = k.split([num_prefix_tokens, num_patches], axis=-2)

    # Original: q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    # Original: k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)
    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    # Original: q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    # Original: k = torch.cat((k_prefix_tokens, k_patches), dim=-2)
    q = paddle.concat([q_prefix_tokens, q_patches], axis=-2)
    k = paddle.concat([k_prefix_tokens, k_patches], axis=-2)

    return q, k


class DINOv3ViTAttention(nn.Layer):
    # Original PyTorch: class DINOv3ViTAttention(nn.Module)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads  # self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads  # self.head_dim = self.embed_dim // self.num_heads

        self.scaling = self.head_dim ** -0.5  # self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout  # self.dropout = config.attention_dropout

        # Original: self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        # Original: self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)
        # Original: self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        # Original: self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.value_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.proj_bias)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        # Original: def forward(self, hidden_states, attention_mask, position_embeddings)
        batch_size, patches, _ = hidden_states.shape

        # Original: query_states = self.q_proj(hidden_states)
        # Original: key_states = self.k_proj(hidden_states)
        # Original: value_states = self.v_proj(hidden_states)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Original: query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        # Original: key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        # Original: value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = query_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        # Original: cos, sin = position_embeddings
        # Original: query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Original: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling

        # Original: if attention_mask is not None: attn_weights = attn_weights + attention_mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Original: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.softmax(attn_weights, axis=-1)
        
        # Critical: Apply dropout during training
        # Original: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        if self.training and self.dropout > 0.0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Original: attn_output = torch.matmul(attn_weights, value)
        # Original: attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, patches, -1])

        # Original: attn_output = self.o_proj(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DINOv3ViTMLP(nn.Layer):
    # Original PyTorch: class DINOv3ViTMLP(nn.Module)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size  # self.intermediate_size = config.intermediate_size

        # Original: self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # Original: self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                 bias_attr=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
                                   bias_attr=config.mlp_bias)
        self.act_fn = nn.GELU()  # Original: self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # Original: return self.down_proj(self.act_fn(self.up_proj(x)))
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTGatedMLP(nn.Layer):
    # Original PyTorch: class DINOv3ViTGatedMLP(nn.Module)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size  # self.intermediate_size = config.intermediate_size

        # Original: self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # Original: self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # Original: self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                    bias_attr=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                 bias_attr=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
                                   bias_attr=config.mlp_bias)
        self.act_fn = nn.GELU()  # Original: self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # Original: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DINOv3ViTLayer(nn.Layer):
    # Original PyTorch: class DINOv3ViTLayer(GradientCheckpointingLayer)
    # This corresponds to the Block class in the original implementation
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Original: self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)  # self.attention = DINOv3ViTAttention(config)

        layer_scale_init = Constant(value=config.layerscale_value)
        self.layer_scale1 = LayerScale(config)  # self.layer_scale1 = DINOv3ViTLayerScale(config)

        # Original: self.drop_path = DINOv3ViTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        if config.drop_path_rate > 0.0:
            self.drop_path = DropPath(config.drop_path_rate)
        else:
            self.drop_path = Identity()

        # Original: self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        # Original: if config.use_gated_mlp: self.mlp = DINOv3ViTGatedMLP(config) else: self.mlp = DINOv3ViTMLP(config)
        if config.use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(config)
        else:
            self.mlp = DINOv3ViTMLP(config)

        self.layer_scale2 = LayerScale(config)  # self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        # Original: def forward(self, hidden_states, attention_mask, position_embeddings)
        
        # Attention with residual connection
        # Original: residual = hidden_states
        # Original: hidden_states = self.norm1(hidden_states)
        # Original: hidden_states, _ = self.attention(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
        # Original: hidden_states = self.layer_scale1(hidden_states)
        # Original: hidden_states = self.drop_path(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # MLP with residual connection
        # Original: residual = hidden_states
        # Original: hidden_states = self.norm2(hidden_states)
        # Original: hidden_states = self.mlp(hidden_states)
        # Original: hidden_states = self.layer_scale2(hidden_states)
        # Original: hidden_states = self.drop_path(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class DINOv3ViTModel(nn.Layer):
    # Original PyTorch: class DINOv3ViTModel(DINOv3ViTPreTrainedModel)
    # PaddleClas wrapper that combines model variants with classification head
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4,
                 qkv_bias=False,  # Deprecated, use query_bias/key_bias/value_bias
                 query_bias=None,
                 key_bias=None,
                 value_bias=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_gated_mlp=False,
                 num_register_tokens=0,
                 layerscale_value=1.0,
                 rope_theta=100.0,
                 pos_embed_shift=None,
                 pos_embed_jitter=None,
                 pos_embed_rescale=None,
                 out_indices=None,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 use_gradient_checkpointing=False,
                 **kwargs):
        super().__init__()
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim
        self.out_indices = out_indices if out_indices is not None else []
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Build config object to match PyTorch DINOv3ViTConfig
        # Original PyTorch uses DINOv3ViTConfig dataclass
        class Config:
            pass
        config = Config()
        config.image_size = img_size  # config.image_size
        config.patch_size = patch_size  # config.patch_size
        config.num_channels = in_chans  # config.num_channels
        config.hidden_size = embed_dim  # config.hidden_size
        config.intermediate_size = int(embed_dim * mlp_ratio)  # config.intermediate_size
        config.num_hidden_layers = depth  # config.num_hidden_layers
        config.num_attention_heads = num_heads  # config.num_attention_heads
        config.hidden_act = 'gelu'  # config.hidden_act
        config.attention_dropout = attn_drop_rate  # config.attention_dropout
        config.initializer_range = 0.02  # config.initializer_range
        config.layer_norm_eps = epsilon  # config.layer_norm_eps
        config.rope_theta = rope_theta  # config.rope_theta

        # Handle bias configuration for backward compatibility
        # If query_bias/key_bias/value_bias are explicitly provided, use them
        # Otherwise, use qkv_bias for all (legacy behavior)
        if query_bias is not None:
            config.query_bias = query_bias
        else:
            config.query_bias = qkv_bias  # config.query_bias
        if key_bias is not None:
            config.key_bias = key_bias
        else:
            config.key_bias = qkv_bias  # config.key_bias
        if value_bias is not None:
            config.value_bias = value_bias
        else:
            config.value_bias = qkv_bias  # config.value_bias

        config.proj_bias = True  # config.proj_bias
        config.mlp_bias = True  # config.mlp_bias
        config.layerscale_value = layerscale_value  # config.layerscale_value
        config.drop_path_rate = drop_path_rate  # config.drop_path_rate
        config.use_gated_mlp = use_gated_mlp  # config.use_gated_mlp
        config.num_register_tokens = num_register_tokens  # config.num_register_tokens
        config.pos_embed_shift = pos_embed_shift  # config.pos_embed_shift
        config.pos_embed_jitter = pos_embed_jitter  # config.pos_embed_jitter
        config.pos_embed_rescale = pos_embed_rescale  # config.pos_embed_rescale
        self.config = config

        # Original: self.embeddings = DINOv3ViTEmbeddings(config)
        # Original: self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)

        # Original: self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.LayerList([
            DINOv3ViTLayer(config) for _ in range(depth)
        ])

        # Original: self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm = nn.LayerNorm(embed_dim, epsilon=epsilon)

        # Classification head (not in original PyTorch model, added for PaddleClas)
        self.head = nn.Linear(embed_dim, class_num) if class_num > 0 else Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Original PyTorch: def _init_weights(self, module)
        # Weight initialization following original implementation
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)  # init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if m.bias is not None:
                zeros_(m.bias)  # init.zeros_(module.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)  # init.zeros_(module.bias) 
            ones_(m.weight)  # init.ones_(module.weight)

    def forward_features(self, x):
        # Original PyTorch: def forward(self, pixel_values, bool_masked_pos)
        # Returns features instead of classification logits
        
        # Original: hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        # Original: position_embeddings = self.rope_embeddings(pixel_values)
        hidden_states = self.embeddings(x)
        position_embeddings = self.rope_embeddings(x)

        backbone_features = []
        if 0 in self.out_indices:
            backbone_features.append(hidden_states)

        # Original: for i, layer_module in enumerate(self.layer):
        #     hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        for idx, layer_module in enumerate(self.layers):
            # Gradient checkpointing support (original uses GradientCheckpointingLayer)
            if self.use_gradient_checkpointing and self.training:
                try:
                    # Original: checkpoint(layer_module, hidden_states, position_embeddings)
                    hidden_states = paddle.distributed.fleet.utils.recompute(
                        layer_module,
                        hidden_states,
                        position_embeddings=position_embeddings,
                        preserve_rng_state=False,
                        use_reentrant=False)
                except:
                    hidden_states = layer_module(
                        hidden_states,
                        position_embeddings=position_embeddings)
            else:
                hidden_states = layer_module(
                    hidden_states,
                    position_embeddings=position_embeddings)
            if (idx + 1) in self.out_indices:
                backbone_features.append(hidden_states)

        # Original: sequence_output = self.norm(hidden_states)
        # Original: pooled_output = sequence_output[:, 0, :]
        sequence_output = self.norm(hidden_states)
        if -1 in self.out_indices or len(self.out_indices) == 0:
            backbone_features.append(sequence_output)

        pooled_output = sequence_output[:, 0, :]  # Extract CLS token

        if len(self.out_indices) > 0:
            return backbone_features  # Multi-level features for backbone mode
        return pooled_output  # Single pooled output for classification

    def forward(self, x):
        # Main forward pass
        # Returns classification logits or multi-level features
        features = self.forward_features(x)
        if isinstance(features, list):  # Backbone mode with out_indices
            return features
        x = self.head(features)  # Classification mode
        return x

    def _convert_hf_state_dict_to_paddle(self, hf_state_dict):
        """
        将 HuggingFace 格式的 state dict 转换为 PaddleClas DINOv3 格式。

        HF 键名格式 -> PaddleClas 键名格式映射：
            dinov3_vit.embeddings.xxx       -> embeddings.xxx
            dinov3_vit.layer.N.xxx          -> layers.N.xxx
            dinov3_vit.norm.xxx             -> norm.xxx
            dinov3_vit.rope_embeddings.xxx  -> rope_embeddings.xxx

        重要：Paddle Linear 层权重存储顺序与 PyTorch 相反
            - PyTorch nn.Linear: weight shape = (out_features, in_features)
            - Paddle nn.Linear: weight shape = (in_features, out_features)
            - 因此需要转置 Linear 层的权重
        """
        import re

        paddle_state_dict = {}
        skipped_keys = []
        conversion_log = []
        transpose_log = []

        # 跳过的键（这些是 PaddleClas 不需要的）
        skip_patterns = [
            "inv_freq",          # RoPE 的 inv_freq 在 forward 中动态计算
            "bias_mask",         # 原始 PyTorch 模型中的 bias_mask
            "local_cls_norm",    # 原始实现中的局部归一化
        ]

        # 需要转置的 Linear 层权重模式
        # 这些层使用 nn.Linear，权重需要转置
        transpose_patterns = [
            r"\.q_proj\.weight$",
            r"\.k_proj\.weight$",
            r"\.v_proj\.weight$",
            r"\.o_proj\.weight$",
            r"\.up_proj\.weight$",
            r"\.down_proj\.weight$",
            r"\.gate_proj\.weight$",
            r"\.head\.weight$",
        ]

        # 不需要转置的层（这些不是标准的 Linear 层）
        no_transpose_patterns = [
            "patch_embeddings.weight",  # Conv2D 权重
            "cls_token",                 # embedding token
            "mask_token",                # embedding token
            "register_tokens",           # embedding token
            "layer_scale",               # LayerScale 参数 (1D)
            ".bias",                     # bias 参数 (1D)
            "norm.weight",               # LayerNorm weight (1D)
            "norm.bias",                 # LayerNorm bias (1D)
        ]

        for hf_key, hf_value in hf_state_dict.items():
            # 1. 移除 HF 模型前缀 "dinov3_vit."
            paddle_key = hf_key.replace("dinov3_vit.", "")

            # 2. 检查是否需要跳过
            should_skip = False
            for pattern in skip_patterns:
                if pattern in paddle_key:
                    should_skip = True
                    skipped_keys.append(hf_key)
                    break
            if should_skip:
                continue

            # 3. HF 的 layer.N -> PaddleClas 的 layers.N
            # 注意：HF 使用 layer，PaddleClas 使用 layers
            original_key = paddle_key
            paddle_key = re.sub(r'\blayer\.(\d+)', r'layers.\1', paddle_key)
            if paddle_key != original_key:
                conversion_log.append(f"  {original_key} -> {paddle_key}")

            # 4. 转换值为 numpy 数组
            if not isinstance(hf_value, np.ndarray):
                if hasattr(hf_value, 'numpy'):
                    hf_value = hf_value.numpy()
                elif hasattr(hf_value, 'cpu'):
                    # 处理 PyTorch tensor
                    hf_value = hf_value.cpu().numpy()
                else:
                    hf_value = np.array(hf_value)

            # 5. 检查是否需要转置 Linear 层权重
            should_transpose = False
            for pattern in transpose_patterns:
                if re.search(pattern, paddle_key):
                    # 检查是否在"不转置"列表中
                    is_excluded = False
                    for exclude_pattern in no_transpose_patterns:
                        if exclude_pattern in paddle_key:
                            is_excluded = True
                            break
                    if not is_excluded and hf_value.ndim == 2:
                        should_transpose = True
                        hf_value = hf_value.T  # 转置权重
                        transpose_log.append(f"  {paddle_key}: {hf_value.shape} -> (转置)")
                    break

            paddle_state_dict[paddle_key] = hf_value

        # 打印转换信息
        if skipped_keys:
            print(f"[HF加载] 跳过 {len(skipped_keys)} 个不需要的键:")
            for key in skipped_keys[:5]:  # 只显示前 5 个
                print(f"        - {key}")
            if len(skipped_keys) > 5:
                print(f"        ... 和其他 {len(skipped_keys) - 5} 个键")

        if conversion_log and len(conversion_log) <= 10:
            print(f"[HF加载] 键名转换示例:")
            for log in conversion_log[:5]:
                print(log)

        if transpose_log:
            print(f"[HF加载] 转置 {len(transpose_log)} 个 Linear 层权重:")
            for log in transpose_log[:5]:
                print(log)
            if len(transpose_log) > 5:
                print(f"        ... 和其他 {len(transpose_log) - 5} 个权重")

        return paddle_state_dict

    def load_hf_pretrained(self, hf_path_or_url):
        """
        从 HuggingFace 格式模型加载预训练权重。

        参数:
            hf_path_or_url: HF 模型目录路径或模型文件路径

        支持的格式:
            - 目录: 包含 model.safetensors 或 pytorch_model.bin 的目录
            - 文件: .safetensors 或 .bin 文件

        注意:
            输入应为已经是 HF 格式的模型（已通过 convert_dinov3_vit_to_hf.py 转换）。
        """
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        # 1. 确定权重文件路径和类型
        weight_file, file_type = self._resolve_hf_weight_path(hf_path_or_url)
        print(f"[HF加载] 从 {weight_file} 加载权重 (格式: {file_type})")

        # 2. 加载 HF state dict
        hf_state_dict = self._load_hf_state_dict(weight_file, file_type)

        # 3. 转换为 PaddleClas 格式
        paddle_state_dict = self._convert_hf_state_dict_to_paddle(hf_state_dict)

        # 4. 检查缺失和多余的键
        self._check_state_dict_match(paddle_state_dict)

        # 5. 加载权重
        self.set_state_dict(paddle_state_dict)
        print(f"[HF加载] 成功加载 {len(paddle_state_dict)} 个参数")

    def _resolve_hf_weight_path(self, hf_path_or_url):
        """解析 HF 权重文件路径和类型"""
        if os.path.isdir(hf_path_or_url):
            # 目录模式：查找 safetensors 或 pytorch_model.bin
            safetensors_path = os.path.join(hf_path_or_url, "model.safetensors")
            if os.path.exists(safetensors_path):
                return safetensors_path, "safetensors"

            bin_path = os.path.join(hf_path_or_url, "pytorch_model.bin")
            if os.path.exists(bin_path):
                return bin_path, "pytorch"

            raise FileNotFoundError(
                f"目录中找不到权重文件 (期望 model.safetensors 或 pytorch_model.bin): {hf_path_or_url}"
            )
        else:
            # 文件模式
            if not os.path.exists(hf_path_or_url):
                raise FileNotFoundError(f"权重文件不存在: {hf_path_or_url}")

            if hf_path_or_url.endswith(".safetensors"):
                return hf_path_or_url, "safetensors"
            elif hf_path_or_url.endswith(".bin") or hf_path_or_url.endswith(".pt"):
                return hf_path_or_url, "pytorch"
            else:
                # 尝试作为 safetensors 处理
                return hf_path_or_url, "safetensors"

    def _load_hf_state_dict(self, weight_file, file_type):
        """加载 HF state dict"""
        if file_type == "safetensors":
            try:
                from safetensors import safe_open
                hf_state_dict = {}
                with safe_open(weight_file, framework="numpy") as f:
                    for key in f.keys():
                        hf_state_dict[key] = f.get_tensor(key)
                return hf_state_dict
            except ImportError:
                raise ImportError("需要安装 safetensors: pip install safetensors")
        else:  # pytorch
            import torch
            state_dict = torch.load(weight_file, map_location="cpu")
            # 转换为 numpy
            return {k: v.numpy() if hasattr(v, 'numpy') else np.array(v)
                    for k, v in state_dict.items()}

    def _check_state_dict_match(self, paddle_state_dict):
        """检查 state dict 与模型参数的匹配情况"""
        model_keys = set(self.state_dict().keys())
        loaded_keys = set(paddle_state_dict.keys())

        missing_keys = sorted(model_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - model_keys)

        if missing_keys:
            print(f"[HF加载] 警告: 模型中有 {len(missing_keys)} 个参数未被加载 (将使用随机初始化):")
            for key in missing_keys[:5]:
                print(f"        - {key}")
            if len(missing_keys) > 5:
                print(f"        ... 和其他 {len(missing_keys) - 5} 个参数")

        if unexpected_keys:
            print(f"[HF加载] 警告: 权重文件中有 {len(unexpected_keys)} 个参数未被使用:")
            for key in unexpected_keys[:5]:
                print(f"        - {key}")
            if len(unexpected_keys) > 5:
                print(f"        ... 和其他 {len(unexpected_keys) - 5} 个参数")

        matched_count = len(model_keys & loaded_keys)
        print(f"[HF加载] 匹配 {matched_count}/{len(model_keys)} 个模型参数")


def _load_pretrained(pretrained, model, model_url, use_ssld=False, hf_pretrained=None):
    # Helper function to load pretrained weights
    # Supports: HuggingFace (hf_pretrained), PaddleClas URL (pretrained=True), or local path
    if hf_pretrained is not None:
        model.load_hf_pretrained(hf_pretrained)
    elif pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type.")


def DINOv3_vits16(pretrained=False, use_ssld=False, hf_pretrained=None, **kwargs):
    # DINOv3 Small model with 16x16 patch size
    # Original HF: facebook/dinov3-vits16-pretrain-lvd1689m
    # HF Config: query_bias=True, key_bias=False, value_bias=True, num_register_tokens=4
    model = DINOv3ViTModel(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        num_register_tokens=4,
        drop_path_rate=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DINOv3_vits16"],
        use_ssld=use_ssld,
        hf_pretrained=hf_pretrained)
    return model


def DINOv3_vits14(pretrained=False, use_ssld=False, hf_pretrained=None, **kwargs):
    # DINOv3 Small model with 14x14 patch size
    # Original HF: facebook/dinov3-small-518
    model = DINOv3ViTModel(
        img_size=518,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DINOv3_vits14"],
        use_ssld=use_ssld,
        hf_pretrained=hf_pretrained)
    return model


def DINOv3_vitb16(pretrained=False, use_ssld=False, hf_pretrained=None, **kwargs):
    # DINOv3 Base model with 16x16 patch size
    # Original HF: facebook/dinov3-base-224-vitsp16
    model = DINOv3ViTModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DINOv3_vitb16"],
        use_ssld=use_ssld,
        hf_pretrained=hf_pretrained)
    return model


def DINOv3_vitb14(pretrained=False, use_ssld=False, hf_pretrained=None, **kwargs):
    # DINOv3 Base model with 14x14 patch size
    # Original HF: facebook/dinov3-base-518
    model = DINOv3ViTModel(
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DINOv3_vitb14"],
        use_ssld=use_ssld,
        hf_pretrained=hf_pretrained)
    return model


def DINOv3_vitg14(pretrained=False, use_ssld=False, hf_pretrained=None, **kwargs):
    # DINOv3 Giant model with 14x14 patch size and Gated MLP (SwiGLU)
    # Original HF: facebook/dinov3-giant-518
    model = DINOv3ViTModel(
        img_size=518,
        patch_size=14,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        use_gated_mlp=True,  # Giant uses SwiGLU activation
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DINOv3_vitg14"],
        use_ssld=use_ssld,
        hf_pretrained=hf_pretrained)
    return model
