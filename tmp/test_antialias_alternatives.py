"""
测试 Paddle 中 antialias 的替代方案
"""
import numpy as np
import paddle
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import cv2

def load_test_image():
    """加载测试图片"""
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    pil_img = Image.open(image_path).convert('RGB')
    return TF.to_tensor(pil_img)

# 方案 1: 使用 PIL resize (PIL 默认有抗锯齿)
def resize_with_pil(image_tensor, size=224):
    """使用 PIL resize - PIL 默认有抗锯齿效果"""
    # 转换为 PIL Image
    pil_img = TF.to_pil_image(image_tensor)
    # Resize - PIL 默认使用 BILINEAR 并有抗锯齿
    resized = pil_img.resize((size, size), Image.BILINEAR)
    # 转回 tensor
    return TF.to_tensor(resized)

# 方案 2: Paddle 的 area 模式
def resize_paddle_area(image_tensor, size=224):
    """使用 Paddle 的 area interpolation (适合缩小)"""
    x = image_tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    resized = paddle.nn.functional.interpolate(
        x,
        size=[size, size],
        mode='area',
        align_corners=False,
        data_format='NCHW'
    )
    return resized.squeeze(0)

# 方案 3: 手动实现高斯模糊 + bilinear
def resize_paddle_gaussian_bilinear(image_tensor, size=224, kernel_size=5, sigma=1.0):
    """先高斯模糊再 bilinear resize"""
    x = image_tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    # 创建高斯核
    def gaussian_kernel(size, sigma):
        """生成 2D 高斯核"""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)
        return kernel.astype(np.float32)
    
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel_tensor = paddle.to_tensor(kernel.reshape(1, 1, kernel_size, kernel_size))
    
    # 对每个通道应用高斯模糊
    kernel_expanded = kernel_tensor.expand([3, 1, kernel_size, kernel_size])
    blurred = paddle.nn.functional.conv2d(
        x, kernel_expanded, padding=kernel_size//2, groups=3
    )
    
    # 然后用 bilinear resize
    resized = paddle.nn.functional.interpolate(
        blurred,
        size=[size, size],
        mode='bilinear',
        align_corners=False,
        data_format='NCHW'
    )
    return resized.squeeze(0)

# 方案 4: 使用 OpenCV 的 INTER_AREA
def resize_opencv_area(image_np, size=224):
    """使用 OpenCV 的 INTER_AREA"""
    hwc = image_np.transpose(1, 2, 0)
    resized = cv2.resize(hwc, (size, size), interpolation=cv2.INTER_AREA)
    return resized.transpose(2, 0, 1)

def compare_methods():
    """对比各种方法"""
    print("=" * 60)
    print("对比 Paddle 中 antialias 的替代方案")
    print("=" * 60)
    
    # 加载测试图片
    image_torch = load_test_image()
    image_np = image_torch.numpy()
    image_paddle = paddle.to_tensor(image_np)
    
    # Reference: Torch resize with antialias=True
    torch_ref = TF.resize(image_torch, [224, 224],
                          interpolation=TF.InterpolationMode.BILINEAR,
                          antialias=True)
    
    print(f"\nReference (Torch antialias=True):")
    print(f"  shape: {torch_ref.shape}, mean: {torch_ref.mean():.6f}")
    print(f"  前4个值: {torch_ref.flatten()[:4].tolist()}")
    
    # 测试各种方案
    methods = [
        ("PIL (BILINEAR, 默认抗锯齿)", lambda: resize_with_pil(image_torch)),
        ("Paddle area 模式", lambda: resize_paddle_area(image_paddle)),
        ("Paddle 高斯+bilinear (k=5,s=1.0)", lambda: resize_paddle_gaussian_bilinear(image_paddle, kernel_size=5, sigma=1.0)),
        ("Paddle 高斯+bilinear (k=7,s=1.5)", lambda: resize_paddle_gaussian_bilinear(image_paddle, kernel_size=7, sigma=1.5)),
        ("OpenCV INTER_AREA", lambda: paddle.to_tensor(resize_opencv_area(image_np))),
    ]
    
    print("\n" + "=" * 60)
    results = []
    for name, method in methods:
        result = method()
        if isinstance(result, paddle.Tensor):
            result_np = result.numpy()
        elif isinstance(result, torch.Tensor):
            result_np = result.numpy()
        else:
            result_np = result
        
        diff = np.abs(torch_ref.numpy() - result_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        results.append((name, max_diff, mean_diff, result_np))
        
        print(f"\n{name}:")
        print(f"  max_diff: {max_diff:.2e}")
        print(f"  mean_diff: {mean_diff:.2e}")
        print(f"  前4个值: {result_np.flatten()[:4].tolist()}")
    
    # 排序并推荐最佳方案
    print("\n" + "=" * 60)
    print("按误差排序 (从小到大):")
    print("=" * 60)
    results.sort(key=lambda x: x[1])
    for i, (name, max_diff, mean_diff, _) in enumerate(results, 1):
        status = "✅" if max_diff < 0.05 else "⚠️" if max_diff < 0.1 else "❌"
        print(f"{i}. {status} {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

if __name__ == "__main__":
    compare_methods()
