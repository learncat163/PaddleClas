#!/usr/bin/env python3
"""
DINOv3 精度对比测试
- HF 使用 HF processor 预处理
- Paddle 使用 torchvision resize (antialias=True) 精确匹配 HF 行为
- 对比两者输出精度
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16, DINOv3_vitb16, DINOv3_vitl16


def load_image(image_path):
    """加载图片"""
    return Image.open(image_path).convert('RGB')


def preprocess_image_paddle(image, size=224):
    """
    PaddleClas 的图像预处理
    - 使用 torchvision resize 匹配 HF 的行为 (antialias=True)
    - 转换为 tensor 并归一化
    """
    # 1. 转换为 tensor (TF.to_tensor 自动做 rescale: [0,255] -> [0,1])
    image_tensor = TF.to_tensor(image)  # (C, H, W), float32, [0, 1]

    # 2. Resize with antialias=True (匹配 HF 的行为)
    resized = TF.resize(image_tensor, [size, size],
                        interpolation=TF.InterpolationMode.BILINEAR,
                        antialias=True)

    # 3. 转换为 numpy
    image_array = resized.numpy()  # (C, H, W), [0, 1]

    # 4. ImageNet 标准化均值和标准差
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    # 5. 标准化: (x - mean) / std
    image_array = (image_array - mean) / std

    # 6. 添加 batch 维度
    image_array = np.expand_dims(image_array, axis=0)  # (1, C, H, W)

    # 7. 转换为 Paddle tensor
    image_tensor = paddle.to_tensor(image_array, dtype='float32')

    return image_tensor

# ========== 原 Torch 版本（已注释） ==========
# def preprocess_image_paddle(image, size=224):
#     """
#     PaddleClas 的图像预处理
#     - 使用 torchvision resize 匹配 HF 的行为 (antialias=True)
#     - 转换为 tensor 并归一化
#     """
#     # 1. 转换为 tensor (TF.to_tensor 自动做 rescale: [0,255] -> [0,1])
#     image_tensor = TF.to_tensor(image)  # (C, H, W), float32, [0, 1]
#
#     # 2. Resize with antialias=True (匹配 HF 的行为)
#     resized = TF.resize(image_tensor, [size, size],
#                         interpolation=TF.InterpolationMode.BILINEAR,
#                         antialias=True)  # 关键: antialias=True
#
#     # 3. 转换为 numpy
#     image_array = resized.numpy()  # (C, H, W), [0, 1]
#
#     # 4. ImageNet 标准化均值和标准差
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
#
#     # 5. 标准化: (x - mean) / std
#     image_array = (image_array - mean) / std
#
#     # 6. 添加 batch 维度
#     image_array = np.expand_dims(image_array, axis=0)  # (1, C, H, W)
#
#     # 7. 转换为 Paddle tensor
#     image_tensor = paddle.to_tensor(image_array, dtype='float32')
#
#     return image_tensor


def compare_outputs(paddle_output, torch_output, model_name):
    """对比输出精度"""
    # 转换为 numpy
    paddle_np = paddle_output.numpy()
    torch_np = torch_output.detach().cpu().numpy()

    # 计算差异
    abs_diff = np.abs(paddle_np - torch_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-8)

    print(f"\n{model_name} 精度分析:")
    print(f"  PaddleClas 前4个值: {paddle_np[0, :4].tolist()}")
    print(f"  HF 前4个值:       {torch_np[0, :4].tolist()}")
    print(f"  绝对误差 (max):   {max_diff:.2e}")
    print(f"  绝对误差 (mean):  {mean_diff:.2e}")
    print(f"  相对误差 (max):   {np.max(rel_diff):.2e}")

    # 检查是否满足 1e-6 精度要求
    if max_diff < 1e-6:
        print(f"  精度: ✅ 满足 1e-6 要求")
    elif max_diff < 1e-5:
        print(f"  精度: ⚠️ 接近 1e-6 (误差 {max_diff:.2e})")
    elif max_diff < 1e-4:
        print(f"  精度: ❌ 误差约 1e-4")
    else:
        print(f"  精度: ❌ 误差 > 1e-4")

    return max_diff


def test_model_comparison(hf_model_path, pdparams_path, paddle_model_class, model_name):
    """对比测试单个模型 - 使用各自独立的预处理"""
    print(f"\n{'='*70}")
    print(f"测试模型: {model_name}")
    print(f"{'='*70}")

    # 1. 加载图片
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    image = load_image(image_path)

    # 2. HF 路径：使用 HF processor 预处理
    processor = AutoImageProcessor.from_pretrained(hf_model_path)
    inputs_hf = processor(images=image, return_tensors="pt")
    pixel_values_hf = inputs_hf['pixel_values']

    print(f"[HF] 输入 shape: {pixel_values_hf.shape}")
    print(f"[HF] 输入 dtype: {pixel_values_hf.dtype}")
    print(f"[HF] 输入范围: [{pixel_values_hf.min():.4f}, {pixel_values_hf.max():.4f}]")

    # 3. Paddle 路径：使用自己的预处理逻辑
    pixel_values_pd = preprocess_image_paddle(image, size=224)

    print(f"[Paddle] 输入 shape: {pixel_values_pd.shape}")
    print(f"[Paddle] 输入 dtype: {pixel_values_pd.dtype}")
    print(f"[Paddle] 输入范围: [{pixel_values_pd.min().item():.4f}, {pixel_values_pd.max().item():.4f}]")

    # 4. 验证预处理结果是否一致
    pixel_values_hf_np = pixel_values_hf.numpy()
    pixel_values_pd_np = pixel_values_pd.numpy()
    preproc_diff = np.max(np.abs(pixel_values_hf_np - pixel_values_pd_np))
    print(f"\n预处理差异: {preproc_diff:.2e}")
    if preproc_diff < 1e-6:
        print("  ✅ 预处理结果一致")
    else:
        print(f"  ⚠️ 预处理结果有微小差异 (max_diff: {preproc_diff:.2e})")

    # 5. HF 模型推理
    hf_model = AutoModel.from_pretrained(hf_model_path)
    hf_model.eval()
    with torch.inference_mode():
        hf_outputs = hf_model(pixel_values_hf)
        hf_pooled = hf_outputs.pooler_output

    # 6. PaddleClas 模型推理
    paddle_model = paddle_model_class(class_num=0, pretrained=False)
    paddle_model.eval()

    # 加载权重
    if os.path.exists(pdparams_path):
        state_dict = paddle.load(pdparams_path)
        paddle_model.set_state_dict(state_dict)
        print(f"已加载权重: {pdparams_path}")
    else:
        print(f"警告: 权重文件不存在 {pdparams_path}")
        return None

    with paddle.no_grad():
        paddle_pooled = paddle_model.forward_features(pixel_values_pd)

    # 7. 对比输出
    max_diff = compare_outputs(paddle_pooled, hf_pooled, model_name)

    return max_diff


def main():
    # 定义模型配置
    models = [
        ("DINOv3_vits16", DINOv3_vits16,
         "/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/",
         "/tmp/dinov3-vits16.pdparams"),
        ("DINOv3_vitb16", DINOv3_vitb16,
         "/home/cao/llm/facebook/dinov3-vitb16-pretrain-lvd1689m/",
         "/tmp/dinov3-vitb16.pdparams"),
        ("DINOv3_vitl16", DINOv3_vitl16,
         "/home/cao/llm/facebook/dinov3-vitl16-pretrain-lvd1689m/",
         "/tmp/dinov3-vitl16.pdparams"),
    ]

    results = []
    for model_name, paddle_model_class, hf_path, pdparams_path in models:
        max_diff = test_model_comparison(hf_path, pdparams_path, paddle_model_class, model_name)
        if max_diff is not None:
            results.append((model_name, max_diff))

    # 汇总
    print(f"\n{'='*70}")
    print(f"精度对比汇总 (使用各自独立的预处理):")
    print(f"{'='*70}")
    for model_name, max_diff in results:
        status = "✅" if max_diff < 1e-6 else "❌"
        print(f"  {status} {model_name}: max_diff = {max_diff:.2e}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
