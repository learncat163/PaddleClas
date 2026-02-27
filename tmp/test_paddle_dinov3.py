"""
DINOv3 PaddleClas 推理测试
使用 HF AutoImageProcessor 进行预处理，确保与 PyTorch 版本完全一致
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torch
from PIL import Image
from transformers import AutoImageProcessor
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16


def load_image(image_path):
    """加载图片"""
    return Image.open(image_path).convert('RGB')


def preprocess_with_hf_processor(image, processor):
    """
    使用 HF AutoImageProcessor 进行预处理
    这确保与 PyTorch 版本的预处理完全一致
    """
    inputs = processor(images=image, return_tensors="pt")
    # 转换为 Paddle tensor
    pixel_values = inputs['pixel_values'][0].numpy()  # (3, H, W)
    pixel_values = np.expand_dims(pixel_values, axis=0)  # (1, 3, H, W)
    return paddle.to_tensor(pixel_values, dtype='float32')


# === 测试代码 ===

# 1. 加载图片
url = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
image = load_image(url)

# 2. 加载 HF processor 和预训练模型路径
pretrained_model_name = "/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

# 3. 使用 HF processor 进行预处理（与 PyTorch 完全一致）
inputs = preprocess_with_hf_processor(image, processor)

# 4. 加载 PaddleClas 模型
model = DINOv3_vits16(
    class_num=0,  # 0 表示只提取特征，不做分类
    hf_pretrained=pretrained_model_name
)
model.eval()

# 5. 推理
with paddle.no_grad():
    pooled_output = model.forward_features(inputs)

# 6. 输出结果
print("Pooled output shape:", pooled_output.shape)
print("数据类型:", pooled_output.dtype)
print("前4个特征值:", pooled_output[0, :4].numpy().tolist())

print("\n" + "="*60)
print("使用 HF AutoImageProcessor 预处理的优点：")
print("  ✓ 与 PyTorch 版本预处理完全一致")
print("  ✓ 消除 PIL resize 在不同框架间的细微差异")
print("  ✓ 确保模型输出精度达到最佳")
