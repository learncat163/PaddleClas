#!/usr/bin/env python3
"""
HuggingFace vs PaddleClas 精度对比脚本
用于 TIPC 测试框架中验证模型迁移精度
"""
import os
import sys
import argparse
import numpy as np

try:
    import paddle
    import torch
    from transformers import AutoImageProcessor, AutoModel
except ImportError as e:
    print(f"Warning: {e}. Some features may not be available.")


def parse_config_file(config_path):
    """从简化配置文件读取参数"""
    config = {}
    current_section = None

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 检测 section
            if line.startswith('===') and line.endswith('==='):
                current_section = line.strip('=').strip()
                continue

            # 解析 key:value
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                config[key] = value

    return config


def parse_args():
    parser = argparse.ArgumentParser()
    # 支持从配置文件读取
    parser.add_argument("--config", type=str,
                        help="从配置文件读取所有参数")

    # 模型参数
    parser.add_argument("--model_name", type=str,
                        help="模型名称，如 DINOv3_vits16")
    parser.add_argument("--hf_model_path", type=str,
                        help="HuggingFace 模型路径")
    parser.add_argument("--pdparams_path", type=str,
                        help="Paddle 权重文件路径")
    parser.add_argument("--image_path", type=str,
                        default="/home/cao/code/self/paddle/dinov3/000000039769.jpg",
                        help="测试图片路径")
    # 精度阈值
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="绝对误差容忍度")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="相对误差容忍度")
    # 模型类映射
    parser.add_argument("--paddle_model_class", type=str,
                        help="Paddle 模型类路径，如 ppcls.arch.backbone.model_zoo.dinov3.DINOv3_vits16")

    args = parser.parse_args()

    # 如果提供了配置文件，从配置文件读取参数
    if args.config:
        config = parse_config_file(args.config)
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                # 类型转换
                if key in ['atol', 'rtol']:
                    setattr(args, key, float(value))
                elif key in ['expected_output_dim', 'image_size', 'batch_size']:
                    setattr(args, key, int(value))
                elif key in ['use_gpu']:
                    setattr(args, key, value.lower() in ['true', '1', 'yes'])
                else:
                    setattr(args, key, value)

    return args


def load_paddle_model(class_path):
    """动态加载 Paddle 模型类"""
    module_path, class_name = class_path.rsplit('.', 1)
    exec(f"from {module_path} import {class_name}")
    return locals()[class_name]


def compare_outputs(paddle_output, torch_output, atol=1e-5, rtol=1e-5):
    """对比输出精度"""
    paddle_np = paddle_output.numpy() if hasattr(paddle_output, 'numpy') else paddle_output
    torch_np = torch_output.detach().cpu().numpy() if hasattr(torch_output, 'detach') else torch_output

    abs_diff = np.abs(paddle_np - torch_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    result = {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "paddle_shape": list(paddle_np.shape),
        "torch_shape": list(torch_np.shape),
        "passed": bool(max_diff < atol)
    }

    return result


def test_precision(args):
    """执行精度对比测试"""
    print(f"\n{'='*70}")
    print(f"测试模型: {args.model_name}")
    print(f"{'='*70}")

    # 动态导入 Paddle 模型
    if args.paddle_model_class:
        paddle_model_class = load_paddle_model(args.paddle_model_class)
    else:
        # 默认映射
        from ppcls.arch.backbone.model_zoo.dinov3 import (
            DINOv3_vits16, DINOv3_vitb16, DINOv3_vitl16
        )
        model_map = {
            "DINOv3_vits16": DINOv3_vits16,
            "DINOv3_vitb16": DINOv3_vitb16,
            "DINOv3_vitl16": DINOv3_vitl16,
        }
        paddle_model_class = model_map.get(args.model_name)
        if paddle_model_class is None:
            raise ValueError(f"未找到模型类: {args.model_name}")

    # 加载图片和预处理
    try:
        from PIL import Image
        image = Image.open(args.image_path).convert('RGB')

        # HF 预处理
        processor = AutoImageProcessor.from_pretrained(args.hf_model_path)
        inputs_hf = processor(images=image, return_tensors="pt")
        pixel_values_hf = inputs_hf['pixel_values']

        # Paddle 预处理 (使用 torchvision 匹配 HF)
        import torchvision.transforms.functional as TF
        image_tensor = TF.to_tensor(image)
        resized = TF.resize(image_tensor, [224, 224],
                           interpolation=TF.InterpolationMode.BILINEAR,
                           antialias=True)
        image_array = resized.numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image_array = (image_array - mean) / std
        image_array = np.expand_dims(image_array, axis=0)
        pixel_values_pd = paddle.to_tensor(image_array, dtype='float32')

    except Exception as e:
        print(f"预处理失败: {e}")
        return {"status": "error", "message": str(e)}

    # HF 推理
    try:
        hf_model = AutoModel.from_pretrained(args.hf_model_path)
        hf_model.eval()
        with torch.inference_mode():
            hf_outputs = hf_model(pixel_values_hf)
            hf_pooled = hf_outputs.pooler_output
    except Exception as e:
        print(f"HF 推理失败: {e}")
        return {"status": "error", "message": f"HF inference failed: {e}"}

    # Paddle 推理
    try:
        paddle_model = paddle_model_class(class_num=0, pretrained=False)
        paddle_model.eval()

        if os.path.exists(args.pdparams_path):
            state_dict = paddle.load(args.pdparams_path)
            paddle_model.set_state_dict(state_dict)
        else:
            return {"status": "error", "message": f"权重文件不存在: {args.pdparams_path}"}

        with paddle.no_grad():
            paddle_pooled = paddle_model.forward_features(pixel_values_pd)
    except Exception as e:
        print(f"Paddle 推理失败: {e}")
        return {"status": "error", "message": f"Paddle inference failed: {e}"}

    # 对比结果
    result = compare_outputs(paddle_pooled, hf_pooled, args.atol, args.rtol)

    print(f"\n精度分析:")
    print(f"  最大误差: {result['max_diff']:.2e}")
    print(f"  平均误差: {result['mean_diff']:.2e}")
    print(f"  Paddle 输出 shape: {result['paddle_shape']}")
    print(f"  HF 输出 shape: {result['torch_shape']}")
    print(f"  状态: {'✅ 通过' if result['passed'] else '❌ 失败'}")

    result["status"] = "success" if result["passed"] else "failed"
    result["model_name"] = args.model_name

    return result


if __name__ == "__main__":
    args = parse_args()
    result = test_precision(args)

    # 输出 JSON 格式结果供 TIPC 解析
    import json
    print("\n=== TIPC RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
