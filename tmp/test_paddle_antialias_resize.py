"""
实现纯 Paddle 的 antialias resize
参考 PyTorch 的实现：在下采样前应用抗锯齿滤波器
"""
import numpy as np
import paddle
import paddle.nn.functional as F
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def create_antialias_kernel(kernel_size=3, sigma=0.5):
    """
    创建抗锯齿滤波器核（高斯核）
    参考 PyTorch 的 antialias 实现
    """
    # 生成高斯核
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def resize_with_antialias_paddle(image_tensor, size=224, interpolation='bilinear'):
    """
    使用 Paddle 实现 antialias resize
    
    Args:
        image_tensor: Paddle tensor, shape (C, H, W)
        size: 目标尺寸
        interpolation: 'bilinear' 或 'bicubic'
    
    Returns:
        Resized tensor, shape (C, size, size)
    """
    C, H, W = image_tensor.shape
    
    # 如果是上采样或尺寸相同，直接返回
    if H <= size and W <= size:
        return F.interpolate(
            image_tensor.unsqueeze(0),
            size=[size, size],
            mode=interpolation,
            align_corners=False,
            data_format='NCHW'
        ).squeeze(0)
    
    # 计算缩放比例
    scale_h = size / H
    scale_w = size / W
    
    # 只在下采样时应用抗锯齿
    # 参考 torchvision: 只有当目标尺寸明显小于原始尺寸时才应用 antialias
    # 一般条件是 scale < 0.8 时应用
    
    # 创建抗锯齿核
    # kernel_size 和 sigma 根据缩放比例动态调整
    # 参考 torchvision 的实现
    if scale_h < 0.8 or scale_w < 0.8:
        # 计算核大小：至少为 3，最大为 11
        # 参考 torchvision: kernel_size 大约与缩放比例成反比
        min_scale = min(scale_h, scale_w)
        kernel_size = max(3, min(11, int(1 / min_scale + 0.5)))
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保是奇数
        
        # sigma 根据 kernel_size 调整
        sigma = (kernel_size - 1) / 4.0  # 这是一般的经验值
        
        # 创建可分离的高斯核（更高效）
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        kernel_1d = np.exp(-(ax**2) / (2. * sigma**2))
        kernel_1d = kernel_1d / np.sum(kernel_1d)
        kernel_1d = kernel_1d.astype(np.float32)
        
        # 转换为 Paddle tensor
        kernel_tensor = paddle.to_tensor(kernel_1d)
        
        # 对图像进行可分离卷积（先水平后垂直）
        x = image_tensor.unsqueeze(0)  # (1, C, H, W)
        
        # 水平方向滤波
        kernel_h = kernel_tensor.reshape([1, 1, 1, kernel_size]).expand([C, 1, 1, kernel_size])
        x = F.conv2d(x, kernel_h, padding=[0, kernel_size//2], groups=C)
        
        # 垂直方向滤波
        kernel_v = kernel_tensor.reshape([1, 1, kernel_size, 1]).expand([C, 1, kernel_size, 1])
        x = F.conv2d(x, kernel_v, padding=[kernel_size//2, 0], groups=C)
    else:
        x = image_tensor.unsqueeze(0)
    
    # 使用指定的插值方法进行 resize
    resized = F.interpolate(
        x,
        size=[size, size],
        mode=interpolation,
        align_corners=False,
        data_format='NCHW'
    )
    
    return resized.squeeze(0)


def test_antialias_resize():
    """测试 antialias resize 的效果"""
    # 加载测试图片
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    pil_img = Image.open(image_path).convert('RGB')
    
    print("=" * 70)
    print("测试 Paddle antialias resize")
    print("=" * 70)
    
    # Reference: Torch resize with antialias=True
    image_tensor_torch = TF.to_tensor(pil_img)
    torch_result = TF.resize(image_tensor_torch, [224, 224],
                             interpolation=TF.InterpolationMode.BILINEAR,
                             antialias=True)
    
    print(f"\nReference (Torch antialias=True):")
    print(f"  shape: {torch_result.shape}, mean: {torch_result.mean():.6f}")
    print(f"  前4个值: {torch_result.flatten()[:4].tolist()}")
    
    # 测试不同的 kernel_size 和 sigma 组合
    configs = [
        ("固定 k=3, s=0.5", 3, 0.5),
        ("固定 k=5, s=1.0", 5, 1.0),
        ("固定 k=7, s=1.5", 7, 1.5),
        ("动态 kernel (推荐)", None, None),
    ]
    
    print(f"\n" + "=" * 70)
    results = []
    
    for name, k, s in configs:
        if k is None:
            # 使用动态 kernel
            image_paddle = paddle.to_tensor(image_tensor_torch.numpy())
            paddle_result = resize_with_antialias_paddle(image_paddle, size=224)
        else:
            # 使用固定 kernel
            image_paddle = paddle.to_tensor(image_tensor_torch.numpy())
            
            C, H, W = image_paddle.shape
            x = image_paddle.unsqueeze(0)
            
            ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
            kernel_1d = np.exp(-(ax**2) / (2. * s**2))
            kernel_1d = kernel_1d / np.sum(kernel_1d)
            kernel_1d = kernel_1d.astype(np.float32)
            
            kernel_tensor = paddle.to_tensor(kernel_1d)
            kernel_h = kernel_tensor.reshape([1, 1, 1, k]).expand([3, 1, 1, k])
            x = F.conv2d(x, kernel_h, padding=[0, k//2], groups=3)
            kernel_v = kernel_tensor.reshape([1, 1, k, 1]).expand([3, 1, k, 1])
            x = F.conv2d(x, kernel_v, padding=[k//2, 0], groups=3)
            
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
    test_antialias_resize()
