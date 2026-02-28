#!/usr/bin/env python3
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch

image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
image = Image.open(image_path).convert('RGB')

print("=" * 70)
print("对比不同 resize 方法的精度")
print("=" * 70)

torch_img = TF.to_tensor(image)
resized_torch = TF.resize(torch_img, [224, 224],
                         interpolation=TF.InterpolationMode.BILINEAR,
                         antialias=True)
torch_np = resized_torch.numpy()

print(f"\n1. Torchvision (antialias=True) - 基准:")
print(f"   Shape: {torch_np.shape}")
print(f"   范围: [{torch_np.min():.6f}, {torch_np.max():.6f}]")
print(f"   前4个像素: {torch_np[0, 0, 0:4]}")

pil_methods = [
    ("NEAREST", Image.NEAREST),
    ("BILINEAR", Image.BILINEAR),
    ("BICUBIC", Image.BICUBIC),
    ("LANCZOS", Image.LANCZOS),
]

for method_name, method in pil_methods:
    resized_pil = image.resize((224, 224), method)
    pil_np = np.array(resized_pil).astype(np.float32) / 255.0
    pil_np = pil_np.transpose(2, 0, 1)
    
    diff = np.max(np.abs(torch_np - pil_np))
    print(f"\n2. PIL {method_name}:")
    print(f"   Shape: {pil_np.shape}")
    print(f"   范围: [{pil_np.min():.6f}, {pil_np.max():.6f}]")
    print(f"   前4个像素: {pil_np[0, 0, 0:4]}")
    print(f"   与 Torch 差异: {diff:.6e}")

print("\n" + "=" * 70)
print("结论: 查看哪个 PIL 方法最接近 Torch antialias=True")
print("=" * 70)
