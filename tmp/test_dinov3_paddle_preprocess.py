#!/usr/bin/env python3
"""
DINOv3 精度对比测试
- HF 使用 HF processor 预处理
- Paddle 使用纯 Paddle/cv2 实现（不依赖 torch）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16, DINOv3_vitb16, DINOv3_vitl16


def load_image(image_path):
    """加载图片"""
    return Image.open(image_path).convert('RGB')


def preprocess_image_paddle_cv2(image, size=224):
    """
    PaddleClas 的图像预处理 - 使用 cv2（纯 Paddle 实现）

    使用 cv2.INTER_AREA 来模拟 torchvision 的 antialias=True 行为
    cv2.INTER_AREA 在缩小图像时使用像素区域关系重采样，
    产生类似抗锯齿的效果
    """
    # 1. 转换为 numpy array (H, W, C), [0, 255]
    image_array = np.array(image).astype(np.float32)

    # 2. Resize using cv2.INTER_AREA
    # INTER_AREA 在缩小图像时使用像素区域关系重采样，
    # 效果接近 torchvision 的 antialias=True
    resized = cv2.resize(image_array, (size, size), interpolation=cv2.INTER_AREA)

    # 3. Rescale: [0, 255] -> [0, 1]
    resized = resized / 255.0

    # 4. 转换为 (C, H, W) 格式
    resized = resized.transpose(2, 0, 1)  # HWC -> CHW

    # 5. ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    normalized = (resized - mean) / std

    # 6. 添加 batch 维度
    normalized = np.expand_dims(normalized, axis=0)  # (1, C, H, W)

    # 7. 转换为 Paddle tensor
    return paddle.to_tensor(normalized, dtype='float32')


def compare_outputs(paddle_output, torch_output, model_name):
    """对比输出精度"""
    paddle_np = paddle_output.numpy()
    torch_np = torch_output.detach().cpu().numpy()

    abs_diff = np.abs(paddle_np - torch_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"\n{model_name} 精度分析:")
    print(f"  PaddleClas 前4个值: {paddle_np[0, :4].tolist()}")
    print(f"  HF 前4个值:       {torch_np[0, :4].tolist()}")
    print(f"  绝对误差 (max):   {max_diff:.2e}")
    print(f"  绝对误差 (mean):  {mean_diff:.2e}")

    if max_diff < 1e-5:
        print(f"  精度: ✅ 优秀")
    elif max_diff < 1e-4:
        print(f"  精度: ⚠️ 良好")
    else:
        print(f"  精度: ❌ 需改进")

    return max_diff


def test_model_comparison(hf_model_path, pdparams_path, paddle_model_class, model_name):
    """对比测试单个模型"""
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
    print(f"[HF] 输入范围: [{pixel_values_hf.min():.4f}, {pixel_values_hf.max():.4f}]")

    # 3. Paddle 路径：使用 cv2 预处理
    pixel_values_pd = preprocess_image_paddle_cv2(image, size=224)

    print(f"[Paddle] 输入 shape: {pixel_values_pd.shape}")
    print(f"[Paddle] 输入范围: [{pixel_values_pd.min().item():.4f}, {pixel_values_pd.max().item():.4f}]")

    # 4. 验证预处理结果
    pixel_values_hf_np = pixel_values_hf.numpy()
    pixel_values_pd_np = pixel_values_pd.numpy()
    preproc_diff = np.max(np.abs(pixel_values_hf_np - pixel_values_pd_np))
    print(f"\n预处理差异: {preproc_diff:.2e}")

    # 5. HF 模型推理
    hf_model = AutoModel.from_pretrained(hf_model_path)
    hf_model.eval()
    with torch.inference_mode():
        hf_outputs = hf_model(pixel_values_hf)
        hf_pooled = hf_outputs.pooler_output

    # 6. PaddleClas 模型推理
    paddle_model = paddle_model_class(class_num=0, pretrained=False)
    paddle_model.eval()

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
    print(f"精度对比汇总 (使用 cv2.INTER_AREA):")
    print(f"{'='*70}")
    for model_name, max_diff in results:
        status = "✅" if max_diff < 1e-4 else "❌"
        print(f"  {status} {model_name}: max_diff = {max_diff:.2e}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
