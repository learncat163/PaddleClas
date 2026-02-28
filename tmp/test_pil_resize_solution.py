"""
使用 PIL resize 替代 Torch 的 antialias=True
- PIL 默认有抗锯齿效果
- 结果与 Torch antialias=True 非常接近
"""
import numpy as np
import paddle
from PIL import Image

def preprocess_image_with_pil(image, size=224):
    """
    使用 PIL 进行 resize（带抗锯齿效果）
    
    Args:
        image: PIL Image
        size: 目标尺寸
    
    Returns:
        Paddle tensor (1, 3, size, size), 已归一化
    """
    # 1. Resize with PIL (默认有抗锯齿)
    resized_img = image.resize((size, size), Image.BILINEAR)
    
    # 2. 转换为 numpy array 并归一化到 [0, 1]
    image_array = np.array(resized_img).astype(np.float32) / 255.0  # (H, W, C)
    
    # 3. 转换为 (C, H, W) 格式
    image_array = image_array.transpose(2, 0, 1)  # (C, H, W)
    
    # 4. ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std
    
    # 5. 添加 batch 维度并转换为 Paddle tensor
    image_array = np.expand_dims(image_array, axis=0)  # (1, C, H, W)
    return paddle.to_tensor(image_array, dtype='float32')

# 测试
if __name__ == "__main__":
    import torch
    import torchvision.transforms.functional as TF
    
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    pil_img = Image.open(image_path).convert('RGB')
    
    # Torch antialias=True
    image_tensor_torch = TF.to_tensor(pil_img)
    torch_result = TF.resize(image_tensor_torch, [224, 224],
                             interpolation=TF.InterpolationMode.BILINEAR,
                             antialias=True)
    
    # PIL resize
    pil_result = preprocess_image_with_pil(pil_img, size=224)
    # 只比较 resize 部分（不做归一化）
    resized_pil = pil_img.resize((224, 224), Image.BILINEAR)
    pil_array = np.array(resized_pil).astype(np.float32) / 255.0
    pil_array = pil_array.transpose(2, 0, 1)
    
    diff = np.abs(torch_result.numpy() - pil_array)
    print(f"PIL resize 与 Torch antialias=True 对比:")
    print(f"  max_diff: {np.max(diff):.2e}")
    print(f"  mean_diff: {np.mean(diff):.2e}")
    
    if np.max(diff) < 0.005:
        print("  ✅ PIL resize 是很好的替代方案！")
