#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastViT Weight Conversion Script
Convert PyTorch weights from timm to PaddlePaddle format.

Usage:
    python tools/fastvit_weight_convert.py --model fastvit_t8 --output_dir ./pretrained/FastViT
"""

import argparse
import os

try:
    import torch
    import timm
except ImportError:
    print("Please install torch and timm: pip install torch timm")
    exit(1)

try:
    import paddle
except ImportError:
    print("Please install paddlepaddle: pip install paddlepaddle")
    exit(1)


# Model name mapping between timm and PaddleClas
MODEL_NAME_MAP = {
    'fastvit_t8': 'FastViT_T8',
    'fastvit_t12': 'FastViT_T12',
    'fastvit_sa12': 'FastViT_SA12',
    'fastvit_sa24': 'FastViT_SA24',
    'fastvit_sa36': 'FastViT_SA36',
    'fastvit_ma36': 'FastViT_MA36',
}


def convert_weight_name(torch_key):
    """
    Convert PyTorch weight key to PaddlePaddle format.

    Args:
        torch_key: PyTorch weight key

    Returns:
        paddle_key: PaddlePaddle weight key
    """
    paddle_key = torch_key

    # Remove 'module.' prefix (from DataParallel)
    paddle_key = paddle_key.replace('module.', '')

    # Handle BN parameter name differences
    # PyTorch: running_mean, running_var, num_batches_tracked
    # Paddle: _mean, _variance, _momentum (num_batches_tracked is not used)
    if 'running_mean' in paddle_key:
        paddle_key = paddle_key.replace('running_mean', '_mean')
    elif 'running_var' in paddle_key:
        paddle_key = paddle_key.replace('running_var', '_variance')
    elif 'num_batches_tracked' in paddle_key:
        # Paddle doesn't use num_batches_tracked, skip it
        return None

    # Handle head layer naming
    # PyTorch: head.fc.weight -> Paddle: head.3.weight (Sequential)
    if 'head.fc' in paddle_key:
        paddle_key = paddle_key.replace('head.fc', 'head.3')

    return paddle_key


def convert_weight_value(torch_value, key, original_key):
    """
    Convert PyTorch weight value to PaddlePaddle format.

    Args:
        torch_value: PyTorch weight tensor
        key: Converted PaddlePaddle weight key
        original_key: Original PyTorch weight key

    Returns:
        paddle_value: PaddlePaddle weight tensor
    """
    # Convert to numpy
    if isinstance(torch_value, torch.Tensor):
        numpy_value = torch_value.cpu().numpy()
    else:
        numpy_value = torch_value

    # Only transpose Linear/FC layers, NOT Conv2D layers
    # In FastViT, mlp.fc1 and mlp.fc2 are Conv2D layers (kernel_size=1)
    # Only head.fc and token_mixer.qkv are true Linear layers
    if any(x in original_key for x in ['head.fc', 'classifier.fc', 'token_mixer.qkv']):
        # Linear layers: PyTorch (out, in) -> Paddle (in, out)
        numpy_value = numpy_value.T

    return numpy_value


def convert_torch_to_paddle(torch_model, paddle_model):
    """
    Convert PyTorch model weights to PaddlePaddle format.

    Args:
        torch_model: PyTorch model
        paddle_model: PaddlePaddle model

    Returns:
        paddle_weights: Dictionary of PaddlePaddle weights
    """
    torch_state_dict = torch_model.state_dict()
    paddle_weights = {}

    # Get Paddle model state dict for reference
    paddle_state_dict = paddle_model.state_dict()

    print(f"Converting {len(torch_state_dict)} weights...")

    converted_count = 0
    skipped_count = 0

    for torch_key, torch_value in torch_state_dict.items():
        paddle_key = convert_weight_name(torch_key)

        # Skip None keys (e.g., num_batches_tracked)
        if paddle_key is None:
            skipped_count += 1
            continue

        # Check if this key exists in Paddle model
        if paddle_key not in paddle_state_dict:
            print(f"  Warning: {paddle_key} not found in Paddle model, skipping")
            skipped_count += 1
            continue

        # Convert weight value
        paddle_value = convert_weight_value(torch_value, paddle_key, torch_key)

        # Check shape compatibility
        expected_shape = paddle_state_dict[paddle_key].shape
        actual_shape = paddle_value.shape

        if expected_shape != actual_shape:
            print(f"  Warning: Shape mismatch for {paddle_key}")
            print(f"    Expected: {expected_shape}")
            print(f"    Got: {actual_shape}")
            print(f"    Skipping...")
            skipped_count += 1
            continue

        paddle_weights[paddle_key] = paddle_value
        converted_count += 1

    print(f"Converted: {converted_count}, Skipped: {skipped_count}")

    return paddle_weights


def download_timm_model(model_name):
    """
    Download FastViT model from timm.

    Args:
        model_name: Model name in timm (e.g., 'fastvit_t8.apple_in1k')

    Returns:
        torch_model: PyTorch model
    """
    print(f"Loading PyTorch model from timm: {model_name}")
    torch_model = timm.create_model(model_name, pretrained=True)
    torch_model.eval()
    return torch_model


def load_paddle_model(model_class_name, num_classes=1000):
    """
    Load PaddlePaddle FastViT model.

    Args:
        model_class_name: Model class name (e.g., 'FastViT_T8')
        num_classes: Number of output classes

    Returns:
        paddle_model: PaddlePaddle model
    """
    # Import FastViT models
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ppcls.arch.backbone.model_zoo.fastvit import (
        FastViT_T8, FastViT_T12, FastViT_SA12,
        FastViT_SA24, FastViT_SA36, FastViT_MA36
    )

    model_dict = {
        'FastViT_T8': FastViT_T8,
        'FastViT_T12': FastViT_T12,
        'FastViT_SA12': FastViT_SA12,
        'FastViT_SA24': FastViT_SA24,
        'FastViT_SA36': FastViT_SA36,
        'FastViT_MA36': FastViT_MA36,
    }

    model_class = model_dict.get(model_class_name)
    if model_class is None:
        raise ValueError(f"Unknown model class: {model_class_name}")

    print(f"Creating PaddlePaddle model: {model_class_name}")
    paddle_model = model_class(num_classes=num_classes)
    paddle_model.eval()
    return paddle_model


def save_paddle_weights(paddle_weights, output_path):
    """
    Save PaddlePaddle weights to file.

    Args:
        paddle_weights: Dictionary of PaddlePaddle weights
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    paddle.save(paddle_weights, output_path)
    print(f"Saved PaddlePaddle weights to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert FastViT weights from PyTorch to PaddlePaddle')
    parser.add_argument('--model', type=str, required=True,
                        choices=['fastvit_t8', 'fastvit_t12', 'fastvit_sa12',
                                'fastvit_sa24', 'fastvit_sa36', 'fastvit_ma36'],
                        help='Model name to convert')
    parser.add_argument('--output_dir', type=str, default='./pretrained/FastViT',
                        help='Output directory for converted weights')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of output classes')

    args = parser.parse_args()

    # Get model names
    timm_model_name = f"{args.model}.apple_in1k"
    paddle_model_name = MODEL_NAME_MAP[args.model]

    # Download PyTorch model
    torch_model = download_timm_model(timm_model_name)

    # Load PaddlePaddle model
    paddle_model = load_paddle_model(paddle_model_name, args.num_classes)

    # Convert weights
    paddle_weights = convert_torch_to_paddle(torch_model, paddle_model)

    # Save weights
    output_path = os.path.join(args.output_dir, f"{paddle_model_name}.pdparams")
    save_paddle_weights(paddle_weights, output_path)

    print("\nConversion completed successfully!")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
