"""
实现纯 Paddle 的 antialias resize (v2)
使用三角形滤波器 (triangle filter) 而不是高斯滤波器
参考 PyTorch 的实现
"""
import numpy as np
import paddle
import paddle.nn.functional as F
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def triangle_filter_1d(size):
    """
    创建三角形滤波器 (PyTorch antialias 使用的滤波器)
    
    对于 bilinear 插值，PyTorch 使用三角形滤波器进行抗锯齿
    滤波器形状是一个等腰三角形
    
    Args:
        size: 滤波器大小 (奇数)
    
    Returns:
        1D 三角形滤波器，sum = 1
    """
    if size < 1:
        size = 1
    if size == 1:
        return np.array([1.0], dtype=np.float32)
    
    # 确保是奇数
    if size % 2 == 0:
        size += 1
    
    # 生成三角形滤波器
    center = (size - 1) / 2.0
    x = np.arange(size)
    kernel = 1.0 - np.abs(x - center) / (size / 2.0)
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def get_antialias_kernel_size(input_size, output_size):
    """
    根据 PyTorch 的规则计算 antialias 滤波器大小
    
    参考 PyTorch 源码中的逻辑：
    - 当 scale >= 0.5 时，kernel_size = 3
    - 当 scale >= 0.33 时，kernel_size = 5
    - 当 scale >= 0.25 时，kernel_size = 7
    - ...
    
    公式：kernel_size ≈ 1 / scale + 1 (然后向上取奇数)
    """
    scale = output_size / input_size
    
    if scale >= 1.0:
        return 1  # 上采样不需要抗锯齿
    
    # PyTorch 的实现：kernel_size = int(2 * np.ceil(1 / scale) + 1)
    # 但实际测试显示是按分段的方式
    kernel_size = int(2 * np.ceil(1 / scale) - 1)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 限制最大值
    if kernel_size > 13:
        kernel_size = 13
    
    return kernel_size


def resize_with_antialias_paddle_v2(image_tensor, size=224):
    """
    使用 Paddle 实现 antialias resize (v2 版本)
    使用三角形滤波器 (triangle filter)，与 PyTorch antialias=True 行为一致
    
    Args:
        image_tensor: Paddle tensor, shape (C, H, W)
        size: 目标尺寸 (正方形)
    
    Returns:
        Resized tensor, shape (C, size, size)
    """
    C, H, W = image_tensor.shape
    
    # 计算缩放比例
    scale_h = size / H
    scale_w = size / W
    min_scale = min(scale_h, scale_w)
    
    # 上采样或 1:1 时直接使用普通插值
    if min_scale >= 1.0:
        return F.interpolate(
            image_tensor.unsqueeze(0),
            size=[size, size],
            mode='bilinear',
            align_corners=False,
            data_format='NCHW'
        ).squeeze(0)
    
    # 计算滤波器大小
    kernel_size_h = get_antialias_kernel_size(H, size)
    kernel_size_w = get_antialias_kernel_size(W, size)
    
    # 使用较大的滤波器尺寸
    kernel_size = max(kernel_size_h, kernel_size_w)
    
    # 创建三角形滤波器
    kernel_1d = triangle_filter_1d(kernel_size)
    kernel_tensor = paddle.to_tensor(kernel_1d)
    
    # 对图像进行可分离卷积（先水平后垂直）
    x = image_tensor.unsqueeze(0)  # (1, C, H, W)
    
    # 水平方向滤波
    kernel_h = kernel_tensor.reshape([1, 1, 1, kernel_size]).expand([C, 1, 1, kernel_size])
    x = F.conv2d(x, kernel_h, padding=[0, kernel_size//2], groups=C)
    
    # 垂直方向滤波
    kernel_v = kernel_tensor.reshape([1, 1, kernel_size, 1]).expand([C, 1, kernel_size, 1])
    x = F.conv2d(x, kernel_v, padding=[kernel_size//2, 0], groups=C)
    
    # 使用 bilinear 插值进行 resize
    resized = F.interpolate(
        x,
        size=[size, size],
        mode='bilinear',
        align_corners=False,
        data_format='NCHW'
    )
    
    return resized.squeeze(0)


def test_paddle_antialias():
    """测试 Paddle antialias resize 的效果"""
    # 加载测试图片
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    pil_img = Image.open(image_path).convert('RGB')
    
    print("=" * 70)
    print("测试 Paddle antialias resize (三角形滤波器版本)")
    print("=" * 70)
    
    # Reference: Torch resize with antialias=True
    image_tensor_torch = TF.to_tensor(pil_img)
    C, H, W = image_tensor_torch.shape
    
    print(f"\n原始图片: shape=({C}, {H}, {W})")
    
    torch_result = TF.resize(image_tensor_torch, [224, 224],
                             interpolation=TF.InterpolationMode.BILINEAR,
                             antialias=True)
    
    print(f"\nReference (Torch antialias=True):")
    print(f"  shape: {torch_result.shape}, mean: {torch_result.mean():.6f}")
    print(f"  前4个值: {torch_result.flatten()[:4].tolist()}")
    
    # 计算应该使用的滤波器大小
    scale = 224 / min(H, W)
    kernel_size = get_antialias_kernel_size(min(H, W), 224)
    print(f"\n缩放比例: {scale:.4f}")
    print(f"计算出的滤波器大小: {kernel_size}")
    print(f"使用的三角形滤波器: {triangle_filter_1d(kernel_size)}")
    
    # 测试不同配置
    configs = [
        ("动态 kernel (推荐)", None),
        ("固定 kernel_size=3", 3),
        ("固定 kernel_size=5", 5),
        ("固定 kernel_size=7", 7),
    ]
    
    print(f"\n" + "=" * 70)
    results = []
    
    for name, fixed_size in configs:
        if fixed_size is None:
            # 使用动态 kernel
            image_paddle = paddle.to_tensor(image_tensor_torch.numpy())
            paddle_result = resize_with_antialias_paddle_v2(image_paddle, size=224)
        else:
            # 使用固定 kernel size
            image_paddle = paddle.to_tensor(image_tensor_torch.numpy())
            
            C, H, W = image_paddle.shape
            x = image_paddle.unsqueeze(0)
            
            kernel_1d = triangle_filter_1d(fixed_size)
            kernel_tensor = paddle.to_tensor(kernel_1d)
            kernel_h = kernel_tensor.reshape([1, 1, 1, fixed_size]).expand([3, 1, 1, fixed_size])
            x = F.conv2d(x, kernel_h, padding=[0, fixed_size//2], groups=3)
            kernel_v = kernel_tensor.reshape([1, 1, fixed_size, 1]).expand([3, 1, fixed_size, 1])
            x = F.conv2d(x, kernel_v, padding=[fixed_size//2, 0], groups=3)
            
            paddle_result = F.interpolate(
                x, size=[224, 224], mode='bilinear',
                align_corners=False, data_format='NCHW'
            ).squeeze(0)
        
        paddle_np = paddle_result.numpy()
        torch_np = torch_result.numpy()
        
        diff = np.abs(torch_np - paddle_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        results.append((name, max_diff, mean_diff))
        
        print(f"\n{name}:")
        print(f"  max_diff: {max_diff:.2e}")
        print(f"  mean_diff: {mean_diff:.2e}")
        print(f"  前4个值: {paddle_np.flatten()[:4].tolist()}")
    
    # 排序
    print(f"\n" + "=" * 70)
    print("按误差排序 (从小到大):")
    print("=" * 70)
    results.sort(key=lambda x: x[1])
    for i, (name, max_diff, mean_diff) in enumerate(results, 1):
        status = "✅" if max_diff < 0.01 else "⚠️" if max_diff < 0.05 else "❌"
        print(f"{i}. {status} {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    return results


if __name__ == "__main__":
    test_paddle_antialias()
