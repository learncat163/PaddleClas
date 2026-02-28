#!/usr/bin/env python3
"""
逐层对比 DINOv3 的输出，分析误差累积
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3ViTModel


def load_image(image_path):
    """加载图片"""
    return Image.open(image_path).convert('RGB')


def create_hf_config(hf_model):
    """从 HF 模型获取配置"""
    config = hf_model.config
    class SimpleConfig:
        pass
    c = SimpleConfig()
    c.image_size = 224
    c.patch_size = 16
    c.num_channels = 3
    c.hidden_size = config.hidden_size
    c.intermediate_size = config.intermediate_size
    c.num_hidden_layers = config.num_hidden_layers
    c.num_attention_heads = config.num_attention_heads
    c.hidden_act = 'gelu'
    c.attention_dropout = config.attention_dropout
    c.initializer_range = config.initializer_range
    c.layer_norm_eps = config.layer_norm_eps
    c.rope_theta = config.rope_theta
    c.query_bias = config.query_bias
    c.key_bias = config.key_bias
    c.value_bias = config.value_bias
    c.proj_bias = True
    c.mlp_bias = True
    c.layerscale_value = 1.0
    c.drop_path_rate = 0.0
    c.use_gated_mlp = getattr(config, 'use_swiglu_fused', False)
    c.num_register_tokens = config.num_register_tokens
    c.pos_embed_shift = None
    c.pos_embed_jitter = None
    c.pos_embed_rescale = None
    return c


def test_single_layer():
    """测试单个 Transformer 层的精度"""
    hf_model_path = "/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/"
    pdparams_path = "/tmp/dinov3-vits16.pdparams"

    # 加载 HF 模型
    hf_model = AutoModel.from_pretrained(hf_model_path)
    hf_model.eval()

    # 创建 Paddle 模型
    config = create_hf_config(hf_model)
    paddle_model = DINOv3ViTModel(
        img_size=224, patch_size=16, in_chans=3, class_num=0,
        embed_dim=config.hidden_size, depth=config.num_hidden_layers,
        num_heads=config.num_attention_heads, mlp_ratio=4,
        query_bias=config.query_bias, key_bias=config.key_bias,
        value_bias=config.value_bias, num_register_tokens=config.num_register_tokens,
        epsilon=config.layer_norm_eps, rope_theta=config.rope_theta
    )
    paddle_model.eval()

    # 加载权重
    state_dict = paddle.load(pdparams_path)
    paddle_model.set_state_dict(state_dict)

    # 准备输入
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    image = load_image(image_path)
    processor = AutoImageProcessor.from_pretrained(hf_model_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values_pt = inputs['pixel_values']
    pixel_values_pd = paddle.to_tensor(pixel_values_pt.numpy(), dtype='float32')

    # 获取 embeddings 输出
    with torch.inference_mode():
        hf_hidden = hf_model.embeddings(pixel_values_pt, bool_masked_pos=None)
        hf_position_embeddings = hf_model.position_embeddings(pixel_values_pt)

    with paddle.no_grad():
        pd_hidden = paddle_model.embeddings(pixel_values_pd)
        pd_position_embeddings = paddle_model.rope_embeddings(pixel_values_pd)

    # 对比 embeddings
    diff = np.abs(pd_hidden.numpy() - hf_hidden.detach().numpy())
    print(f"Embeddings 差异 (max): {np.max(diff):.2e}")
    print(f"Embeddings 差异 (mean): {np.mean(diff):.2e}")

    # 对比 position embeddings
    cos_pd, sin_pd = pd_position_embeddings
    cos_pt, sin_pt = hf_position_embeddings
    cos_diff = np.abs(cos_pd.numpy() - cos_pt.detach().numpy())
    sin_diff = np.abs(sin_pd.numpy() - sin_pt.detach().numpy())
    print(f"Position Cos 差异 (max): {np.max(cos_diff):.2e}")
    print(f"Position Sin 差异 (max): {np.max(sin_diff):.2e}")

    # 逐层测试
    print("\n逐层精度分析:")
    for i, (hf_layer, pd_layer) in enumerate(zip(hf_model.encoder.layers, paddle_model.layers)):
        with torch.inference_mode():
            hf_layer_out = hf_layer(hf_hidden, position_embeddings=hf_position_embeddings)[0]
        with paddle.no_grad():
            pd_layer_out = pd_layer(pd_hidden, position_embeddings=pd_position_embeddings)

        diff = np.abs(pd_layer_out.numpy() - hf_layer_out.detach().numpy())
        print(f"Layer {i}: max_diff = {np.max(diff):.2e}, mean_diff = {np.mean(diff):.2e}")

        # 下一层的输入
        hf_hidden = hf_layer_out
        pd_hidden = pd_layer_out

    # 最终 norm 输出
    with torch.inference_mode():
        hf_final = hf_model.layernorm(hf_hidden)
    with paddle.no_grad():
        pd_final = paddle_model.norm(pd_hidden)

    diff = np.abs(pd_final.numpy() - hf_final.detach().numpy())
    print(f"\nFinal Norm 差异 (max): {np.max(diff):.2e}")
    print(f"Final Norm 差异 (mean): {np.mean(diff):.2e}")

    # CLS token 输出
    hf_pooled = hf_final[:, 0, :]
    pd_pooled = pd_final[:, 0, :]
    diff = np.abs(pd_pooled.numpy() - hf_pooled.detach().numpy())
    print(f"\nCLS Token 差异 (max): {np.max(diff):.2e}")
    print(f"CLS Token 差异 (mean): {np.mean(diff):.2e}")


if __name__ == "__main__":
    test_single_layer()
