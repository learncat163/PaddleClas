"""
对比不同 resize 实现方式
"""
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def resize_torch(image_tensor, size=224):
    """Torch resize with antialias=True"""
    return TF.resize(image_tensor, [size, size],
                     interpolation=TF.InterpolationMode.BILINEAR,
                     antialias=True)

def resize_opencv(image_np, size=224):
    """OpenCV resize with INTER_AREA (for downsampling)"""
    # Convert (C, H, W) to (H, W, C)
    image_hwc = image_np.transpose(1, 2, 0)
    # Resize
    resized = cv2.resize(image_hwc, (size, size), interpolation=cv2.INTER_AREA)
    # Convert back to (C, H, W)
    return resized.transpose(2, 0, 1)

def resize_paddle_tensor(image_tensor, size=224):
    """使用 Paddle 的 bilinear interpolate（带 antialias 近似）"""
    import paddle
    
    # 添加 batch 维度: (C, H, W) -> (1, C, H, W)
    x = image_tensor.unsqueeze(0)
    
    # 先做轻微模糊来模拟 antialias 效果
    # 创建一个 3x3 高斯核
    kernel_size = 3
    sigma = 0.5
    gaussian_kernel = paddle.zeros([1, 1, kernel_size, kernel_size])
    for i in range(kernel_size):
        for j in range(kernel_size):
            x_val = (i - kernel_size // 2) ** 2
            y_val = (j - kernel_size // 2) ** 2
            gaussian_kernel[0, 0, i, j] = np.exp(-(x_val + y_val) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    # 对每个通道应用高斯模糊
    blurred = paddle.nn.functional.conv2d(
        x, 
        gaussian_kernel.expand([3, 1, kernel_size, kernel_size]),
        padding=1,
        groups=3
    )
    
    # 使用 bilinear 插值 resize
    resized = paddle.nn.functional.interpolate(
        blurred,
        size=[size, size],
        mode='bilinear',
        align_corners=False,
        data_format='NCHW'
    )
    
    # 移除 batch 维度
    return resized.squeeze(0)

# 测试
if __name__ == "__main__":
    # 加载测试图片
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    pil_img = Image.open(image_path).convert('RGB')
    
    # 转换为 tensor
    image_tensor = TF.to_tensor(pil_img)
    
    # 各种 resize 方法
    result_torch = resize_torch(image_tensor)
    result_opencv = resize_opencv(image_tensor.numpy())
    
    # Paddle tensor 方法
    import paddle
    image_paddle = paddle.to_tensor(image_tensor.numpy())
    result_paddle = resize_paddle_tensor(image_paddle)
    
    # 对比
    print(f"Torch antialias: shape={result_torch.shape}, mean={result_torch.mean():.6f}")
    print(f"OpenCV INTER_AREA: shape={result_opencv.shape}, mean={result_opencv.mean():.6f}")
    print(f"Paddle blur+bilinear: shape={result_paddle.shape}, mean={result_paddle.mean().item():.6f}")
    
    # 计算差异
    diff_opencv = np.abs(result_torch.numpy() - result_opencv).max()
    diff_paddle = np.abs(result_torch.numpy() - result_paddle.numpy()).max()
    
    print(f"\n与 Torch antialias 差异:")
    print(f"  OpenCV: {diff_opencv:.2e}")
    print(f"  Paddle: {diff_paddle:.2e}")
