"""检查预处理是否一致"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torch
from PIL import Image
from transformers import AutoImageProcessor

def preprocess_image_paddle(image, size=224):
    """PaddleClas 预处理"""
    w, h = image.size
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return paddle.to_tensor(image, dtype='float32')

url = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
pretrained_model_name = "/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/"

print("=" * 60)
print("检查预处理是否一致")
print("=" * 60)

# 加载图片
image = Image.open(url).convert('RGB')

# 1. HF 预处理
print("\n[1] HF 预处理:")
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
inputs_hf = processor(images=image, return_tensors="pt")
pixel_values_hf = inputs_hf['pixel_values'][0]  # [3, 224, 224]
print(f"    形状: {pixel_values_hf.shape}")
print(f"    数据类型: {pixel_values_hf.dtype}")
print(f"    值范围: [{pixel_values_hf.min():.4f}, {pixel_values_hf.max():.4f}]")
print(f"    [0,0,0] 像素值 (R,G,B): {pixel_values_hf[:, 0, 0].tolist()}")

# 2. PaddleClas 预处理
print("\n[2] PaddleClas 预处理:")
inputs_paddle = preprocess_image_paddle(image, size=224)
pixel_values_paddle = inputs_paddle[0]  # [3, 224, 224]
print(f"    形状: {pixel_values_paddle.shape}")
print(f"    数据类型: {pixel_values_paddle.dtype}")
print(f"    值范围: [{pixel_values_paddle.min().item():.4f}, {pixel_values_paddle.max().item():.4f}]")
print(f"    [0,0,0] 像素值 (R,G,B): {pixel_values_paddle[:, 0, 0].numpy().tolist()}")

# 3. 对比
print("\n[3] 对比:")
pixel_values_hf_np = pixel_values_hf.numpy()
pixel_values_paddle_np = pixel_values_paddle.numpy()
diff = np.abs(pixel_values_hf_np - pixel_values_paddle_np)
print(f"    最大差异: {np.max(diff):.6f}")
print(f"    平均差异: {np.mean(diff):.6f}")
print(f"    是否一致: {np.allclose(pixel_values_hf_np, pixel_values_paddle_np, atol=1e-5)}")

if np.max(diff) > 0.01:
    print("\n    ⚠ 预处理有差异，可能影响最终输出")
    # 显示几个不同位置的对比
    for y, x in [(0, 0), (100, 100), (223, 223)]:
        hf_val = pixel_values_hf_np[:, y, x]
        paddle_val = pixel_values_paddle_np[:, y, x]
        print(f"    位置[{y},{x}] HF: {hf_val.tolist()}")
        print(f"    位置[{y},{x}] Paddle: {paddle_val.tolist()}")
