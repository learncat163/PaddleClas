"""对比 PaddleClas DINOv3 和 PyTorch DINOv3 的输出"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def preprocess_image_paddle(image, size=224):
    """
    PaddleClas 预处理，与 HF ImageProcessor 完全一致
    HF DINOv3 使用的是直接 resize 到目标大小，不是短边 resize + center crop
    """
    # 直接 resize 到目标大小（使用 BILINEAR 插值，resample=2）
    image = image.resize((size, size), Image.BILINEAR)

    # 转换为数组并归一化到 [0, 1]
    image = np.array(image).astype(np.float32) / 255.0

    # ImageNet 标准化（与 HF 一致）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    # 添加 batch 维度
    image = np.expand_dims(image, axis=0)

    return paddle.to_tensor(image, dtype='float32')

# === 测试参数 ===
url = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
pretrained_model_name = "/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/"

print("=" * 60)
print("对比 PaddleClas DINOv3 和 PyTorch DINOv3 输出")
print("=" * 60)

# === PyTorch DINOv3 ===
print("\n[1] 运行 PyTorch DINOv3...")
image = load_image(url)
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model_torch = AutoModel.from_pretrained(pretrained_model_name)
inputs_torch = processor(images=image, return_tensors="pt").to(model_torch.device)
with torch.inference_mode():
    outputs_torch = model_torch(**inputs_torch)
pooled_torch = outputs_torch.pooler_output[0, :4].detach().cpu().numpy()
print(f"    PyTorch 前4个值: {pooled_torch.tolist()}")

# === PaddleClas DINOv3 ===
print("\n[2] 运行 PaddleClas DINOv3...")
inputs_paddle = preprocess_image_paddle(image, size=224)
model_paddle = DINOv3_vits16(
    class_num=0,
    hf_pretrained=pretrained_model_name
)
model_paddle.eval()
with paddle.no_grad():
    pooled_paddle = model_paddle.forward_features(inputs_paddle)
pooled_paddle = pooled_paddle[0, :4].numpy()
print(f"    PaddleClas 前4个值: {pooled_paddle.tolist()}")

# === 对比 ===
print("\n[3] 对比结果:")
diff = np.abs(pooled_torch - pooled_paddle)
print(f"    绝对误差: {diff.tolist()}")
print(f"    最大误差: {np.max(diff):.6f}")
print(f"    相对误差(%): {(diff / (np.abs(pooled_torch) + 1e-8) * 100).tolist()}")

if np.max(diff) < 0.01:
    print("\n    ✓ 输出基本一致!")
elif np.max(diff) < 0.1:
    print("\n    ⚠ 输出有差异，可能需要进一步调试")
else:
    print("\n    ✗ 输出差异较大，需要检查模型实现")
