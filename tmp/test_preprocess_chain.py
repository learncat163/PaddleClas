"""
对比完整的预处理链路
"""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor

# 加载测试图片
image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
pil_img = Image.open(image_path).convert('RGB')

# HF processor 预处理
processor = AutoImageProcessor.from_pretrained("/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/")
inputs_hf = processor(images=pil_img, return_tensors="pt")
pixel_values_hf = inputs_hf['pixel_values']

# 方法1: Torch resize with antialias=True
image_tensor = TF.to_tensor(pil_img)
torch_resized = TF.resize(image_tensor, [224, 224],
                          interpolation=TF.InterpolationMode.BILINEAR,
                          antialias=True)
torch_array = torch_resized.numpy()

# 方法2: PIL resize
pil_resized_img = pil_img.resize((224, 224), Image.BILINEAR)
pil_array = np.array(pil_resized_img).astype(np.float32) / 255.0
pil_array = pil_array.transpose(2, 0, 1)

# 归一化
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

torch_normalized = (torch_array - mean) / std
pil_normalized = (pil_array - mean) / std

print("=" * 60)
print("完整预处理链路对比")
print("=" * 60)

# 1. Resize 阶段对比
diff_resize = np.abs(torch_array - pil_array)
print(f"\n1. Resize 阶段 (归一化前):")
print(f"   max_diff: {np.max(diff_resize):.2e}")

# 2. 归一化后对比
diff_normalized = np.abs(torch_normalized - pil_normalized)
print(f"\n2. 归一化后:")
print(f"   max_diff: {np.max(diff_normalized):.2e}")

# 3. 与 HF processor 对比
diff_torch_hf = np.abs(torch_normalized - pixel_values_hf.numpy())
diff_pil_hf = np.abs(pil_normalized - pixel_values_hf.numpy())

print(f"\n3. 与 HF processor 对比:")
print(f"   Torch vs HF: max_diff = {np.max(diff_torch_hf):.2e}")
print(f"   PIL vs HF:   max_diff = {np.max(diff_pil_hf):.2e}")

# 4. 检查 HF processor 的实现方式
print(f"\n4. HF processor 信息:")
print(f"   do_resize: {processor.do_resize}")
print(f"   do_center_crop: {processor.do_center_crop}")
print(f"   do_rescale: {processor.do_rescale}")
print(f"   do_normalize: {processor.do_normalize}")
print(f"   resample: {processor.resample}")  # PIL resampling filter
print(f"   size: {processor.size}")

# 5. HF processor 的 resize 参数
if hasattr(processor, 'image_processor'):
    ip = processor.image_processor
    print(f"\n5. ImageProcessor 详细参数:")
    print(f"   do_resize: {ip.do_resize}")
    print(f"   resample: {ip.resample}")
    print(f"   do_antialias: {getattr(ip, 'do_antialias', 'N/A')}")
