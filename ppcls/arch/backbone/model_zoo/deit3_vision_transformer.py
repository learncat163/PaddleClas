# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was heavily based on https://github.com/facebookresearch/deit
# reference: https://arxiv.org/abs/2204.07118 (DeiT III: Revenge of ViT)

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant

from .vision_transformer import (
    VisionTransformer, PatchEmbed, Attention, Mlp,
    Identity, trunc_normal_, zeros_, DropPath, to_2tuple
)

from ....utils.save_load import load_dygraph_pretrain


class LayerScale(nn.Layer):
    """LayerScale on tensors with channels in last-dim (BLC format)."""
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = self.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=Constant(init_values))
        self.add_parameter("gamma", self.gamma)

    def forward(self, x):
        return x * self.gamma


class DeiT3Block(nn.Layer):
    """Transformer block with LayerScale support for DeiT3."""
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 init_values=None):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, type):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            proj_bias=True)
        
        # LayerScale for attention branch
        self.ls1 = LayerScale(dim, init_values) if init_values else Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else Identity()
        
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, type):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       bias=True)
        
        # LayerScale for MLP branch
        self.ls2 = LayerScale(dim, init_values) if init_values else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DeiT3VisionTransformer(VisionTransformer):
    """DeiT3 Vision Transformer."""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 no_embed_class=False,
                 init_values=None,
                 **kwargs):
        # Initialize parent class
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            class_num=class_num,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            epsilon=epsilon,
            **kwargs)
        
        # DeiT3 specific parameters
        self.no_embed_class = no_embed_class
        
        # Rebuild position embedding if no_embed_class is True
        if no_embed_class:
            # Position embedding does NOT include class token
            self.pos_embed = self.create_parameter(
                shape=(1, self.patch_embed.num_patches, self.embed_dim),
                default_initializer=zeros_)
            self.add_parameter("pos_embed", self.pos_embed)
        
        # Replace blocks with DeiT3Block that supports LayerScale
        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.LayerList([
            DeiT3Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                init_values=init_values
            ) for i in range(depth)
        ])
        
        # Re-initialize weights
        trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        if self.no_embed_class:
            # DeiT3: Add position embedding first, then concatenate class token
            x = x + self.pos_embed
            cls_tokens = self.cls_token.expand((B, -1, -1)).astype(x.dtype)
            x = paddle.concat((cls_tokens, x), axis=1)
        else:
            # Standard DeiT/ViT: Concatenate class token first, then add position embedding
            cls_tokens = self.cls_token.expand((B, -1, -1)).astype(x.dtype)
            x = paddle.concat((cls_tokens, x), axis=1)
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


# Model configuration URLs (placeholder, to be updated with actual URLs)
MODEL_URLS = {
    "DeiT3_small_patch16_224": "",
    "DeiT3_base_patch16_384": "",
    "DeiT3_small_patch16_384": "",
    "DeiT3_large_patch16_384": "",
    "DeiT3_base_patch16_224": "",
    "DeiT3_huge_patch14_224": "",
    "DeiT3_medium_patch16_224": "",
    "DeiT3_large_patch16_224": "",
}

__all__ = list(MODEL_URLS.keys())


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def DeiT3_small_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-small model @ 224x224."""
    model = DeiT3VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        no_embed_class=True,
        init_values=1e-6,
        qkv_bias=True,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_small_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT3_base_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-base model @ 384x384."""
    model = DeiT3VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        no_embed_class=True,
        init_values=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_base_patch16_384"],
        use_ssld=use_ssld)
    return model


def DeiT3_small_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-small model @ 384x384."""
    model = DeiT3VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        no_embed_class=True,
        init_values=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_small_patch16_384"],
        use_ssld=use_ssld)
    return model


def DeiT3_large_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-large model @ 384x384."""
    model = DeiT3VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        no_embed_class=True,
        init_values=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_large_patch16_384"],
        use_ssld=use_ssld)
    return model


def DeiT3_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-base model @ 224x224."""
    model = DeiT3VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        no_embed_class=True,
        init_values=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_base_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT3_huge_patch14_224(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-huge model @ 224x224."""
    model = DeiT3VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        no_embed_class=True,
        init_values=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_huge_patch14_224"],
        use_ssld=use_ssld)
    return model


def DeiT3_medium_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-medium model @ 224x224."""
    model = DeiT3VisionTransformer(
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        no_embed_class=True,
        init_values=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_medium_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT3_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    """DeiT3-large model @ 224x224."""
    model = DeiT3VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        no_embed_class=True,
        init_values=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT3_large_patch16_224"],
        use_ssld=use_ssld)
    return model
