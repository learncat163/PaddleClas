#!/usr/bin/env python3
"""
基本依赖信息如下：
paddlepaddle==2.6.2
numpy==1.24.4



DINOv3 HuggingFace 模型转换为 PaddlePaddle 格式

用法:
    # 从 HF safetensors 文件转换
    python transfer_dinov3_vit_hf_to_pd.py -i /path/to/model.safetensors -o output.pdparams

    # 从 HF 目录转换（自动查找 model.safetensors 或 pytorch_model.bin）
    python transfer_dinov3_vit_hf_to_pd.py -i /path/to/hf_model_dir -o output.pdparams

注意:
    - 输入应为 HuggingFace 格式的 DINOv3 模型
    - 输出为 PaddlePaddle .pdparams 格式
    - 转换后会自动验证参数完整性
    - 转换后的模型可直接通过 DINOv3ViTModel 加载
"""

import argparse
import os
import re

import numpy as np


def convert_hf_state_dict_to_paddle(hf_state_dict, verbose=True):
    """
    将 HuggingFace 格式的 state dict 转换为 PaddleClas DINOv3 格式。

    HF 键名格式 -> PaddleClas 键名格式映射：
        dinov3_vit.embeddings.xxx       -> embeddings.xxx
        dinov3_vit.layer.N.xxx          -> layers.N.xxx
        dinov3_vit.norm.xxx             -> norm.xxx
        dinov3_vit.rope_embeddings.xxx  -> rope_embeddings.xxx

    重要：Paddle Linear 层权重存储顺序与 PyTorch 相反
        - PyTorch nn.Linear: weight shape = (out_features, in_features)
        - Paddle nn.Linear: weight shape = (in_features, out_features)
        - 因此需要转置 Linear 层的权重
    """
    paddle_state_dict = {}
    skipped_keys = []
    conversion_log = []
    transpose_log = []

    # 跳过的键（这些是 PaddleClas 不需要的）
    skip_patterns = [
        "inv_freq",          # RoPE 的 inv_freq 在 forward 中动态计算
        "bias_mask",         # 原始 PyTorch 模型中的 bias_mask
        "local_cls_norm",    # 原始实现中的局部归一化
    ]

    # 需要转置的 Linear 层权重模式
    transpose_patterns = [
        r"\.q_proj\.weight$",
        r"\.k_proj\.weight$",
        r"\.v_proj\.weight$",
        r"\.o_proj\.weight$",
        r"\.up_proj\.weight$",
        r"\.down_proj\.weight$",
        r"\.gate_proj\.weight$",
        r"\.head\.weight$",
    ]

    # 不需要转置的层
    no_transpose_patterns = [
        "patch_embeddings.weight",  # Conv2D 权重
        "cls_token",                 # embedding token
        "mask_token",                # embedding token
        "register_tokens",           # embedding token
        "layer_scale",               # LayerScale 参数 (1D)
        ".bias",                     # bias 参数 (1D)
        "norm.weight",               # LayerNorm weight (1D)
        "norm.bias",                 # LayerNorm bias (1D)
    ]

    for hf_key, hf_value in hf_state_dict.items():
        # 1. 移除 HF 模型前缀 "dinov3_vit."
        paddle_key = hf_key.replace("dinov3_vit.", "")

        # 2. 检查是否需要跳过
        should_skip = False
        for pattern in skip_patterns:
            if pattern in paddle_key:
                should_skip = True
                skipped_keys.append(hf_key)
                break
        if should_skip:
            continue

        # 3. HF 的 layer.N -> PaddleClas 的 layers.N
        original_key = paddle_key
        paddle_key = re.sub(r'\blayer\.(\d+)', r'layers.\1', paddle_key)
        if paddle_key != original_key:
            conversion_log.append(f"  {original_key} -> {paddle_key}")

        # 4. 转换值为 numpy 数组
        if not isinstance(hf_value, np.ndarray):
            if hasattr(hf_value, 'numpy'):
                hf_value = hf_value.numpy()
            elif hasattr(hf_value, 'cpu'):
                hf_value = hf_value.cpu().numpy()
            else:
                hf_value = np.array(hf_value)

        # 5. 检查是否需要转置 Linear 层权重
        for pattern in transpose_patterns:
            if re.search(pattern, paddle_key):
                is_excluded = False
                for exclude_pattern in no_transpose_patterns:
                    if exclude_pattern in paddle_key:
                        is_excluded = True
                        break
                if not is_excluded and hf_value.ndim == 2:
                    hf_value = hf_value.T
                    transpose_log.append(f"  {paddle_key}: {hf_value.shape} -> (转置)")
                break

        paddle_state_dict[paddle_key] = hf_value

    # 打印转换信息
    if verbose:
        print(f"\n[转换信息]")
        if skipped_keys:
            print(f"  跳过 {len(skipped_keys)} 个不需要的键:")
            for key in skipped_keys[:5]:
                print(f"    - {key}")
            if len(skipped_keys) > 5:
                print(f"    ... 和其他 {len(skipped_keys) - 5} 个键")

        if conversion_log and len(conversion_log) <= 10:
            print(f"  键名转换示例:")
            for log in conversion_log[:5]:
                print(log)

        if transpose_log:
            print(f"  转置 {len(transpose_log)} 个 Linear 层权重:")
            for log in transpose_log[:5]:
                print(log)
            if len(transpose_log) > 5:
                print(f"    ... 和其他 {len(transpose_log) - 5} 个权重")

        print(f"  转换后参数数量: {len(paddle_state_dict)}")

    return paddle_state_dict


def resolve_hf_weight_path(hf_path_or_url):
    """解析 HF 权重文件路径和类型"""
    if os.path.isdir(hf_path_or_url):
        safetensors_path = os.path.join(hf_path_or_url, "model.safetensors")
        if os.path.exists(safetensors_path):
            return safetensors_path, "safetensors"

        bin_path = os.path.join(hf_path_or_url, "pytorch_model.bin")
        if os.path.exists(bin_path):
            return bin_path, "pytorch"

        raise FileNotFoundError(
            f"目录中找不到权重文件 (期望 model.safetensors 或 pytorch_model.bin): {hf_path_or_url}"
        )
    else:
        if not os.path.exists(hf_path_or_url):
            raise FileNotFoundError(f"权重文件不存在: {hf_path_or_url}")

        if hf_path_or_url.endswith(".safetensors"):
            return hf_path_or_url, "safetensors"
        elif hf_path_or_url.endswith(".bin") or hf_path_or_url.endswith(".pt"):
            return hf_path_or_url, "pytorch"
        else:
            return hf_path_or_url, "safetensors"


def load_hf_state_dict(weight_file, file_type):
    """加载 HF state dict"""
    print(f"\n[加载] 从 {weight_file} 加载权重 (格式: {file_type})")

    if file_type == "safetensors":
        try:
            from safetensors import safe_open
            hf_state_dict = {}
            with safe_open(weight_file, framework="numpy") as f:
                for key in f.keys():
                    hf_state_dict[key] = f.get_tensor(key)
            print(f"  加载了 {len(hf_state_dict)} 个参数")
            return hf_state_dict
        except ImportError:
            raise ImportError("需要安装 safetensors: pip install safetensors")
    else:  # pytorch
        import torch
        state_dict = torch.load(weight_file, map_location="cpu")
        hf_state_dict = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v)
                         for k, v in state_dict.items()}
        print(f"  加载了 {len(hf_state_dict)} 个参数")
        return hf_state_dict


def save_paddle_params(paddle_state_dict, output_path):
    """保存 PaddlePaddle 参数"""
    import paddle

    output_path = str(output_path)
    if not output_path.endswith('.pdparams'):
        output_path += '.pdparams'

    paddle.save(paddle_state_dict, output_path)
    print(f"\n[保存] 成功保存到: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def validate_conversion(paddle_state_dict):
    """验证转换结果"""
    print(f"\n[验证] 检查转换后的参数")

    # 检查关键参数是否存在
    expected_patterns = [
        "embeddings.cls_token",
        "embeddings.patch_embeddings.weight",
        "norm.weight",
        "norm.bias",
    ]

    missing = []
    for pattern in expected_patterns:
        found = any(pattern in k for k in paddle_state_dict.keys())
        if not found:
            missing.append(pattern)

    if missing:
        print(f"  警告: 缺少预期参数: {missing}")
    else:
        print(f"  关键参数检查通过")

    # 打印参数统计
    param_shapes = {k: v.shape for k, v in paddle_state_dict.items()}
    print(f"  参数总数: {len(param_shapes)}")

    # 打印一些示例参数
    print(f"\n  示例参数:")
    for i, (k, v) in enumerate(list(paddle_state_dict.items())[:5]):
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="将 HuggingFace DINOv3 模型转换为 PaddlePaddle 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入路径: HF 模型目录、.safetensors 或 .bin 文件"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出文件路径 (.pdparams)"
    )

    args = parser.parse_args()

    output_path = args.output

    print(f"=" * 60)
    print(f"DINOv3 HF -> PaddlePaddle 模型转换工具")
    print(f"=" * 60)
    print(f"输入: {args.input}")
    print(f"输出: {output_path}")

    # 1. 解析权重文件路径
    weight_file, file_type = resolve_hf_weight_path(args.input)

    # 2. 加载 HF state dict
    hf_state_dict = load_hf_state_dict(weight_file, file_type)

    # 3. 转换为 PaddleClas 格式
    paddle_state_dict = convert_hf_state_dict_to_paddle(hf_state_dict, verbose=True)

    # 4. 验证转换结果（强制执行）
    validate_conversion(paddle_state_dict)

    # 5. 保存 PaddlePaddle 参数
    save_paddle_params(paddle_state_dict, output_path)

    print(f"\n" + "=" * 60)
    print(f"转换完成!")
    print(f"=" * 60)
    print(f"\n使用方法:")
    print(f"  import paddle")
    print(f"  from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16")
    print(f"  ")
    print(f"  model = DINOv3_vits16(pretrained=False)")
    print(f"  model.set_state_dict(paddle.load('{output_path}'))")


if __name__ == "__main__":
    main()
