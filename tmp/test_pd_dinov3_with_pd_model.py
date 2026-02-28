#!/usr/bin/env python3
"""
DINOv3 PaddleClas 推理测试
使用转换后的 pdparams 模型进行推理
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torchvision.transforms.functional as TF
from PIL import Image
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16, DINOv3_vitb16, DINOv3_vitl16


def load_image(image_path):
    """加载图片"""
    return Image.open(image_path).convert('RGB')


def preprocess_image(image, size=224):
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
                        antialias=True)  # 关键: antialias=True

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


def test_model(model_class, model_name, pdparams_path, inputs):
    """
    测试指定模型的推理

    Args:
        model_class: 模型类 (如 DINOv3_vits16)
        model_name: 模型名称 (用于打印)
        pdparams_path: pdparams 权重文件路径
        inputs: 输入 tensor
    """
    # 检查文件是否存在
    if not os.path.exists(pdparams_path):
        print(f"\n{'='*60}")
        print(f"跳过模型: {model_name}")
        print(f"权重文件不存在: {pdparams_path}")
        print(f"{'='*60}")
        return None

    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")

    # 创建模型
    model = model_class(class_num=0, pretrained=False)
    model.eval()

    # 加载权重
    state_dict = paddle.load(pdparams_path)
    model.set_state_dict(state_dict)
    print(f"已加载权重: {pdparams_path}")

    # 推理
    with paddle.no_grad():
        pooled_output = model.forward_features(inputs)

    # 输出结果
    print(f"推理结果:")
    print(f"  Pooled output shape: {pooled_output.shape}")
    print(f"  数据类型: {pooled_output.dtype}")
    print(f"  前4个特征值: {pooled_output[0, :4].numpy().tolist()}")

    return pooled_output


def main():
    # 1. 加载图片
    image_path = "docs/images/inference_deployment/whl_demo.jpg"
    image = load_image(image_path)

    # 2. 预处理
    inputs = preprocess_image(image, size=224)
    print(f"输入 shape: {inputs.shape}")

    # 3. 定义要测试的模型列表
    models_to_test = [
        (DINOv3_vits16, "DINOv3_vits16", "/tmp/dinov3-vits16.pdparams"),
        (DINOv3_vitb16, "DINOv3_vitb16", "/tmp/dinov3-vitb16.pdparams"),
        (DINOv3_vitl16, "DINOv3_vitl16", "/tmp/dinov3-vitl16.pdparams"),
    ]

    # 4. 依次测试每个模型
    results = []
    for model_class, model_name, pdparams_path in models_to_test:
        result = test_model(model_class, model_name, pdparams_path, inputs)
        if result is not None:
            results.append((model_name, result))

    # 5. 汇总
    print(f"\n{'='*60}")
    print(f"测试完成! 成功测试 {len(results)} 个模型")
    for model_name, _ in results:
        print(f"  - {model_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
