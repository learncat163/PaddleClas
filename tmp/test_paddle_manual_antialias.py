#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModel
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16

def preprocess_paddle_manual_antialias(image, size=224):
    """手动实现 antialias 的预处理"""
    image_array = np.array(image).astype(np.float32)

    # 计算缩放比例
    h, w = image_array.shape[0], image_array.shape[1]
    scale = min(size / h, size / w)

    # 如果是缩小，先进行高斯模糊
    if scale < 1.0:
        # 根据 torchvision 的实现计算 sigma
        sigma = max(0.1, 0.5 * (1 / scale - 1))
        # 使用 scipy 的高斯模糊
        blurred_array = gaussian_filter(image_array, sigma=[sigma, sigma, 0], mode='reflect', truncate=6.0)
        image_array = blurred_array.astype(np.float32)

    # 转为 Paddle tensor [C, H, W]
    image_tensor = paddle.to_tensor(image_array.transpose(2, 0, 1), dtype='float32')

    # Resize
    image_tensor = image_tensor.unsqueeze(0)
    resized = F.interpolate(image_tensor, size=[size, size], mode='bilinear', align_corners=False)
    resized = resized.squeeze(0)
    resized = resized / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    normalized = (resized.numpy() - mean) / std
    normalized = np.expand_dims(normalized, axis=0)
    return paddle.to_tensor(normalized, dtype='float32')

# 测试
image = Image.open("/home/cao/code/self/paddle/dinov3/000000039769.jpg").convert('RGB')
processor = AutoImageProcessor.from_pretrained("/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/")
inputs_hf = processor(images=image, return_tensors="pt")
hf_model = AutoModel.from_pretrained("/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/")
hf_model.eval()
with torch.inference_mode():
    hf_out = hf_model(inputs_hf['pixel_values']).pooler_output

inputs_pd = preprocess_paddle_manual_antialias(image)
pd_model = DINOv3_vits16(class_num=0, pretrained=False)
state_dict = paddle.load("/tmp/dinov3-vits16.pdparams")
pd_model.set_state_dict(state_dict)
pd_model.eval()
with paddle.no_grad():
    pd_out = pd_model.forward_features(inputs_pd)

hf_np = hf_out.detach().cpu().numpy()
pd_np = pd_out.numpy()
diff = np.abs(pd_np - hf_np)
print(f'=== 手动 antialias (scipy gaussian) 精度 ===')
print(f'max_diff: {np.max(diff):.2e}')
print(f'mean_diff: {np.mean(diff):.2e}')
print(f'Paddle: {pd_np[0, :4]}')
print(f'HF:    {hf_np[0, :4]}')

# 检查预处理差异
hf_input = inputs_hf['pixel_values'].numpy()
pd_input = inputs_pd.numpy()
preproc_diff = np.max(np.abs(hf_input - pd_input))
print(f'预处理差异: {preproc_diff:.2e}')
