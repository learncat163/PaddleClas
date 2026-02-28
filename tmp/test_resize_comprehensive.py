#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torchvision.transforms.functional as TF
from PIL import Image
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16, DINOv3_vitb16


def preprocess_pil(image, size=224):
    resized_img = image.resize((size, size), Image.BILINEAR)
    image_array = np.array(resized_img).astype(np.float32) / 255.0
    image_array = image_array.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std
    image_array = np.expand_dims(image_array, axis=0)
    return paddle.to_tensor(image_array, dtype='float32')


def preprocess_torch(image, size=224):
    image_tensor = TF.to_tensor(image)
    resized = TF.resize(image_tensor, [size, size],
                        interpolation=TF.InterpolationMode.BILINEAR,
                        antialias=True)
    image_array = resized.numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std
    image_array = np.expand_dims(image_array, axis=0)
    return paddle.to_tensor(image_array, dtype='float32')


def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)


def extract_features(model, image, preprocess_fn):
    tensor = preprocess_fn(image)
    with paddle.no_grad():
        features = model.forward_features(tensor)
    return features.numpy()[0]


def test_single_image_variants(model, image_path):
    print("\n" + "=" * 70)
    print("测试 1: 同一图片的不同变换")
    print("=" * 70)
    
    image = Image.open(image_path).convert('RGB')
    
    feat_pil = extract_features(model, image, preprocess_pil)
    feat_torch = extract_features(model, image, preprocess_torch)
    
    cos_sim = cosine_similarity(feat_pil, feat_torch)
    l2_dist = np.linalg.norm(feat_pil - feat_torch)
    max_diff = np.max(np.abs(feat_pil - feat_torch))
    
    print(f"PIL vs TorchVision:")
    print(f"  余弦相似度: {cos_sim:.10f}")
    print(f"  L2 距离: {l2_dist:.6f}")
    print(f"  最大绝对差: {max_diff:.6f}")
    
    variants = [
        ("原图", image),
        ("旋转90度", image.rotate(90, expand=True)),
        ("水平翻转", image.transpose(Image.FLIP_LEFT_RIGHT)),
        ("裁剪中心", image.crop((50, 50, image.width-50, image.height-50))),
    ]
    
    print("\n不同变换下的特征差异:")
    for name, var_img in variants:
        feat_pil_var = extract_features(model, var_img, preprocess_pil)
        feat_torch_var = extract_features(model, var_img, preprocess_torch)
        
        cos_sim_var = cosine_similarity(feat_pil_var, feat_torch_var)
        l2_var = np.linalg.norm(feat_pil_var - feat_torch_var)
        
        print(f"  {name}:")
        print(f"    余弦相似度: {cos_sim_var:.10f}")
        print(f"    L2 距离: {l2_var:.6f}")


def test_classification_confidence(model, image_path):
    print("\n" + "=" * 70)
    print("测试 2: 分类置信度影响（使用特征范数模拟）")
    print("=" * 70)
    
    image = Image.open(image_path).convert('RGB')
    
    feat_pil = extract_features(model, image, preprocess_pil)
    feat_torch = extract_features(model, image, preprocess_torch)
    
    norm_pil = np.linalg.norm(feat_pil)
    norm_torch = np.linalg.norm(feat_torch)
    norm_diff_pct = abs(norm_pil - norm_torch) / norm_torch * 100
    
    print(f"特征向量范数:")
    print(f"  PIL: {norm_pil:.6f}")
    print(f"  TorchVision: {norm_torch:.6f}")
    print(f"  差异百分比: {norm_diff_pct:.4f}%")
    
    top_k = 10
    top_pil = np.argsort(feat_pil)[-top_k:][::-1]
    top_torch = np.argsort(feat_torch)[-top_k:][::-1]
    
    print(f"\nTop-{top_k} 最大激活维度:")
    print(f"  PIL:   {top_pil.tolist()}")
    print(f"  Torch: {top_torch.tolist()}")
    
    overlap = len(set(top_pil) & set(top_torch))
    print(f"  重叠数量: {overlap}/{top_k} ({overlap/top_k*100:.1f}%)")


def test_retrieval_stability():
    print("\n" + "=" * 70)
    print("测试 3: 检索稳定性（噪声敏感度）")
    print("=" * 70)
    
    model = DINOv3_vits16(class_num=0, pretrained=False)
    model.eval()
    state_dict = paddle.load("/tmp/dinov3-vits16.pdparams")
    model.set_state_dict(state_dict)
    
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    image = Image.open(image_path).convert('RGB')
    
    np.random.seed(42)
    
    n_trials = 5
    cos_sims = []
    
    for i in range(n_trials):
        img_array = np.array(image)
        noise = np.random.randn(*img_array.shape) * 5
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_array)
        
        feat_pil_orig = extract_features(model, image, preprocess_pil)
        feat_pil_noisy = extract_features(model, noisy_img, preprocess_pil)
        
        feat_torch_orig = extract_features(model, image, preprocess_torch)
        feat_torch_noisy = extract_features(model, noisy_img, preprocess_torch)
        
        cos_pil = cosine_similarity(feat_pil_orig, feat_pil_noisy)
        cos_torch = cosine_similarity(feat_torch_orig, feat_torch_noisy)
        
        cos_sims.append((cos_pil, cos_torch))
    
    print(f"原图 vs 加噪图的余弦相似度 (噪声 σ=5, {n_trials} 次):")
    for i, (cos_pil, cos_torch) in enumerate(cos_sims):
        print(f"  试验 {i+1}: PIL={cos_pil:.8f}, Torch={cos_torch:.8f}, diff={abs(cos_pil-cos_torch):.8f}")
    
    avg_cos_pil = np.mean([c[0] for c in cos_sims])
    avg_cos_torch = np.mean([c[1] for c in cos_sims])
    
    print(f"\n平均相似度:")
    print(f"  PIL: {avg_cos_pil:.8f}")
    print(f"  Torch: {avg_cos_torch:.8f}")
    print(f"  平均差异: {abs(avg_cos_pil - avg_cos_torch):.8f}")


def test_model_sizes():
    print("\n" + "=" * 70)
    print("测试 4: 不同模型尺寸的影响")
    print("=" * 70)
    
    models_config = [
        ("vit-small", DINOv3_vits16, "/tmp/dinov3-vits16.pdparams"),
        ("vit-base", DINOv3_vitb16, "/tmp/dinov3-vitb16.pdparams"),
    ]
    
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    image = Image.open(image_path).convert('RGB')
    
    for name, model_class, weight_path in models_config:
        if not os.path.exists(weight_path):
            print(f"\n{name}: 权重文件不存在，跳过")
            continue
            
        model = model_class(class_num=0, pretrained=False)
        model.eval()
        state_dict = paddle.load(weight_path)
        model.set_state_dict(state_dict)
        
        feat_pil = extract_features(model, image, preprocess_pil)
        feat_torch = extract_features(model, image, preprocess_torch)
        
        cos_sim = cosine_similarity(feat_pil, feat_torch)
        l2_dist = np.linalg.norm(feat_pil - feat_torch)
        max_diff = np.max(np.abs(feat_pil - feat_torch))
        
        print(f"\n{name} (维度: {len(feat_pil)}):")
        print(f"  余弦相似度: {cos_sim:.10f}")
        print(f"  L2 距离: {l2_dist:.6f}")
        print(f"  最大绝对差: {max_diff:.6f}")


def main():
    print("=" * 70)
    print("DINOv3 Resize 方法实际影响深度分析")
    print("=" * 70)
    
    model = DINOv3_vits16(class_num=0, pretrained=False)
    model.eval()
    state_dict = paddle.load("/tmp/dinov3-vits16.pdparams")
    model.set_state_dict(state_dict)
    print("已加载模型权重")
    
    image_path = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
    
    test_single_image_variants(model, image_path)
    test_classification_confidence(model, image_path)
    test_retrieval_stability()
    test_model_sizes()
    
    print("\n" + "=" * 70)
    print("结论汇总")
    print("=" * 70)
    print("关键观察指标:")
    print("  1. 余弦相似度: >0.9999 表示特征方向几乎一致")
    print("  2. L2 距离: <0.1 表示差异可忽略")
    print("  3. Top-K 重叠: >90% 表示检索结果一致")
    print("  4. 噪声稳定性: 两种方法应表现相似")
    print("=" * 70)


if __name__ == "__main__":
    main()
