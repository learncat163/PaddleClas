#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import paddle
import torchvision.transforms.functional as TF
from PIL import Image
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16


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


def main():
    print("=" * 70)
    print("DINOv3 实际任务测试: PIL vs TorchVision Resize")
    print("=" * 70)
    
    model = DINOv3_vits16(class_num=0, pretrained=False)
    model.eval()
    state_dict = paddle.load("/tmp/dinov3-vits16.pdparams")
    model.set_state_dict(state_dict)
    print("已加载模型权重\n")
    
    test_images = [
        "/home/cao/code/self/paddle/dinov3/000000039769.jpg",
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}")
            continue
            
        print(f"测试图片: {os.path.basename(img_path)}")
        image = Image.open(img_path).convert('RGB')
        
        feat_pil = extract_features(model, image, preprocess_pil)
        feat_torch = extract_features(model, image, preprocess_torch)
        
        abs_diff = np.abs(feat_pil - feat_torch)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        cos_sim = cosine_similarity(feat_pil, feat_torch)
        
        print(f"  特征维度: {feat_pil.shape}")
        print(f"  PIL 特征范围: [{feat_pil.min():.4f}, {feat_pil.max():.4f}]")
        print(f"  Torch 特征范围: [{feat_torch.min():.4f}, {feat_torch.max():.4f}]")
        print(f"  绝对误差 (max): {max_diff:.6f}")
        print(f"  绝对误差 (mean): {mean_diff:.6f}")
        print(f"  余弦相似度: {cos_sim:.8f}")
        print(f"  特征距离 (L2): {np.linalg.norm(feat_pil - feat_torch):.6f}")
        print()
    
    print("=" * 70)
    print("图像检索模拟测试")
    print("=" * 70)
    
    test_dir = "/home/cao/code/self/paddle/dinov3/"
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))][:5]
    
    if len(image_files) < 2:
        print("测试图片不足，跳过检索测试")
    else:
        print(f"使用 {len(image_files)} 张图片进行检索测试\n")
        
        images = [Image.open(os.path.join(test_dir, f)).convert('RGB') for f in image_files]
        
        feats_pil = [extract_features(model, img, preprocess_pil) for img in images]
        feats_torch = [extract_features(model, img, preprocess_torch) for img in images]
        
        query_idx = 0
        print(f"查询图片: {image_files[query_idx]}\n")
        
        print("使用 PIL resize:")
        sims_pil = [cosine_similarity(feats_pil[query_idx], feats_pil[i]) 
                    for i in range(len(feats_pil))]
        for i, sim in enumerate(sims_pil):
            print(f"  {image_files[i]}: {sim:.6f}")
        
        print("\n使用 TorchVision resize:")
        sims_torch = [cosine_similarity(feats_torch[query_idx], feats_torch[i]) 
                      for i in range(len(feats_torch))]
        for i, sim in enumerate(sims_torch):
            print(f"  {image_files[i]}: {sim:.6f}")
        
        print("\n相似度差异:")
        for i in range(len(sims_pil)):
            diff = abs(sims_pil[i] - sims_torch[i])
            print(f"  {image_files[i]}: {diff:.6f}")
        
        rank_pil = sorted(range(len(sims_pil)), key=lambda i: sims_pil[i], reverse=True)
        rank_torch = sorted(range(len(sims_torch)), key=lambda i: sims_torch[i], reverse=True)
        
        print(f"\nPIL 排序: {[image_files[i] for i in rank_pil]}")
        print(f"Torch 排序: {[image_files[i] for i in rank_torch]}")
        
        if rank_pil == rank_torch:
            print("✅ 检索排序完全一致")
        else:
            print("⚠️ 检索排序有差异")
    
    print("\n" + "=" * 70)
    print("结论分析")
    print("=" * 70)
    print("1. 特征向量差异: 观察 max_diff 和 mean_diff")
    print("2. 余弦相似度: 接近 1.0 说明方向一致")
    print("3. 检索排序: 实际应用中的影响")
    print("=" * 70)


if __name__ == "__main__":
    main()
