import os
import sys
import re
import argparse

sys.path.insert(0, "/home/cao/code/github/PaddleClas/raw-transformer-dinov3")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    import torch
    import paddle
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install paddlepaddle torch")
    exit(1)


HUB_MODELS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}

HF_MODEL_NAMES = {
    "vits16": "facebook/dinov3-vits16",
    "vitb16": "facebook/dinov3-vitb16",
}


def rename_hf_key_to_paddle(hf_key):
    new_key = hf_key.replace("dinov3_vit.", "")

    mappings = {
        "embeddings.cls_token": "cls_token",
        "embeddings.mask_token": "mask_token",
        "embeddings.register_tokens": "register_tokens",
        "embeddings.patch_embeddings": "patch_embeddings",
        "rope_embeddings.inv_freq": "rope_embeddings.inv_freq",
    }

    for hf_pattern, paddle_suffix in mappings.items():
        if new_key.startswith(hf_pattern):
            return new_key.replace(hf_pattern, paddle_suffix)

    new_key = re.sub(r'^layer\.(\d+)\.', r'layers.\1.', new_key)

    return new_key


def convert_hf_to_paddle(hf_model_path, output_path, model_name):
    print(f"Loading HuggingFace model from {hf_model_path}...")

    sys.path.insert(0, "/home/cao/code/github/PaddleClas/raw-transformer-dinov3")
    from modeling_dinov3_vit import DINOv3ViTModel, DINOv3ViTConfig

    config = DINOv3ViTConfig.from_pretrained(hf_model_path)
    hf_model = DINOv3ViTModel.from_pretrained(hf_model_path)
    hf_model.eval()

    hf_state_dict = hf_model.state_dict()
    paddle_state_dict = {}

    print("Converting weights...")
    for hf_key, hf_tensor in hf_state_dict.items():
        paddle_key = rename_hf_key_to_paddle(hf_key)

        if "inv_freq" in hf_key:
            continue

        hf_value = hf_tensor.detach().cpu().numpy()
        paddle_state_dict[paddle_key] = hf_value

    print(f"Converted {len(paddle_state_dict)} parameters")

    paddle_model_params = {}
    for key, value in paddle_state_dict.items():
        paddle_model_params[key] = paddle.Tensor(value)

    paddle.save(paddle_model_params, output_path)
    print(f"Saved PaddlePaddle weights to {output_path}")

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(HUB_MODELS.keys()), default="vits16",
                        help="Model name to convert")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for converted weights")
    parser.add_argument("--from-hf", action="store_true",
                        help="Load from HuggingFace Hub")
    args = parser.parse_args()

    model_name = args.model
    hf_model_name = HF_MODEL_NAMES[model_name]

    if args.output is None:
        args.output = f"tmp/dinov3_{model_name}_pdparams.pdparams"

    if args.from_hf:
        print(f"Loading from HuggingFace Hub: {hf_model_name}...")
        config = convert_hf_to_paddle(hf_model_name, args.output, model_name)

    print(f"\nConversion complete!")
    print(f"Model config:")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - num_hidden_layers: {config.num_hidden_layers}")
    print(f"  - num_attention_heads: {config.num_attention_heads}")
    print(f"  - patch_size: {config.patch_size}")
    print(f"  - num_register_tokens: {config.num_register_tokens}")
    print(f"  - use_gated_mlp: {config.use_gated_mlp}")
    print(f"\nTo use the converted weights:")
    print(f'  model = DINOv3ViTModel(...)')
    print(f'  state_dict = paddle.load("{args.output}")')
    print(f'  model.set_state_dict(state_dict)')


if __name__ == "__main__":
    main()
