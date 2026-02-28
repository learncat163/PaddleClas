# DINOv3 Antialias Resize 测试总结

## 背景

DINOv3 模型的预处理需要使用带抗锯齿 (antialias) 的图像缩放。
PyTorch 的 `torchvision.transforms.functional.resize` 支持 `antialias=True` 参数，
但 PaddlePaddle 的 `interpolate` 不支持此参数。

## 测试目标

找到纯 Paddle 的替代方案，或找到不依赖 Torch 的可行替代方案。

## 测试结果

### Resize 阶段对比 (与 Torch antialias=True 的差异)

| 方案 | max_diff | mean_diff | 说明 |
|------|----------|-----------|------|
| **PIL BILINEAR** | **3.72e-03** | **1.08e-03** | ✅ 最佳 |
| PIL BICUBIC | 1.01e-01 | 6.08e-03 | 误差较大 |
| PIL LANCZOS | 1.60e-01 | 9.08e-03 | 误差较大 |
| Paddle area | 1.42e-01 | 7.89e-03 | 误差较大 |
| Paddle bilinear (无AA) | 4.11e-01 | 2.18e-02 | 误差很大 |
| Torch bilinear (无AA) | 4.11e-01 | 2.18e-02 | 误差很大 |

### 完整模型精度对比 (使用 PIL BILINEAR 预处理)

| 模型 | max_diff | 评价 |
|------|----------|------|
| DINOv3_vits16 | 8.44e-03 | ⚠️ 可接受 |
| DINOv3_vitb16 | 1.76e-02 | ⚠️ 可接受 |
| DINOv3_vitl16 | 1.83e-02 | ⚠️ 可接受 |

预处理差异: 1.63e-02

### 对比：使用 Torch antialias=True 的精度

| 模型 | max_diff | 评价 |
|------|----------|------|
| DINOv3_vits16 | ~1e-06 | ✅ 完美 |
| DINOv3_vitb16 | ~1e-06 | ✅ 完美 |
| DINOv3_vitl16 | ~1e-06 | ✅ 完美 |

## 结论

1. **纯 Paddle 的 antialias 实现无法完美复现 PyTorch 的行为**
   - 尝试了高斯滤波、三角形滤波等多种方法，误差都较大
   
2. **PIL BILINEAR 是最佳的替代方案**
   - Resize 阶段误差仅 3.72e-03
   - 不依赖 Torch
   - 完整模型误差在 8e-03 ~ 2e-02 之间，可接受

3. **如果对精度要求极高（1e-6 级别），必须使用 Torch**

## 推荐实现

```python
from PIL import Image
import numpy as np

def preprocess_image_pil(image, size=224):
    """
    使用 PIL resize 替代 Torch antialias
    """
    # PIL resize (默认有抗锯齿效果)
    resized_img = image.resize((size, size), Image.BILINEAR)
    
    # 转换为 numpy
    image_array = np.array(resized_img).astype(np.float32) / 255.0
    image_array = image_array.transpose(2, 0, 1)  # HWC -> CHW
    
    # 归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std
    
    return image_array
```

## 测试文件

- `tmp/test_dinov3_precision.py` - 完整模型精度测试 (使用 PIL)
- `tmp/test_antialias_summary.py` - 所有方案对比总结
