"""
直接对比 PIL resize 和 Torch antialias resize 的差异
"""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# 加载测试图片
image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
pil_img = Image.open(image_path).convert('RGB')

# 方法 1: Torch resize with antialias=True
image_tensor = TF.to_tensor(pil_img)
torch_resized = TF.resize(image_tensor, [224, 224],
                          interpolation=TF.InterpolationMode.BILINEAR,
                          antialias=True)

# 方法 2: PIL resize (BILINEAR)
pil_resized_img = pil_img.resize((224, 224), Image.BILINEAR)
pil_array = np.array(pil_resized_img).astype(np.float32) / 255.0
pil_array = pil_array.transpose(2, 0, 1)  # HWC -> CHW

# 对比差异
torch_np = torch_resized.numpy()
diff = np.abs(torch_np - pil_array)

print("=" * 60)
print("PIL resize vs Torch antialias=True 直接对比")
print("=" * 60)
print(f"\nTorch antialias: shape={torch_np.shape}, mean={torch_np.mean():.6f}")
print(f"PIL BILINEAR:    shape={pil_array.shape}, mean={pil_array.mean():.6f}")
print(f"\n差异分析:")
print(f"  max_diff:  {np.max(diff):.2e}")
print(f"  mean_diff: {np.mean(diff):.2e}")
print(f"  median_diff: {np.median(diff):.2e}")
print(f"\n前4个像素值对比:")
print(f"  Torch: {torch_np.flatten()[:4].tolist()}")
print(f"  PIL:   {pil_array.flatten()[:4].tolist()}")
print(f"  差异:  {diff.flatten()[:4].tolist()}")

# 可视化差异分布
print(f"\n差异分布:")
print(f"  < 0.001: {np.sum(diff < 0.001)} / {diff.size}")
print(f"  0.001-0.01: {np.sum((diff >= 0.001) & (diff < 0.01))} / {diff.size}")
print(f"  >= 0.01: {np.sum(diff >= 0.01)} / {diff.size}")
