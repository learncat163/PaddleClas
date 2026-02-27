import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import numpy as np
from ppcls.arch.backbone.model_zoo import dinov3
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits14

def test_embeddings_initialization():
    print("=" * 60)
    print("测试 Embeddings 初始化")
    print("=" * 60)
    
    model = DINOv3_vits14(class_num=10)
    
    cls_token = model.embeddings.cls_token
    mask_token = model.embeddings.mask_token
    register_tokens = model.embeddings.register_tokens
    
    print(f"✓ cls_token shape: {cls_token.shape}")
    print(f"   cls_token std: {float(cls_token.std()):.4f} (期望 ~1.0)")
    
    print(f"✓ mask_token shape: {mask_token.shape}")
    print(f"   mask_token all zeros: {bool((mask_token == 0).all())}")
    
    print(f"✓ register_tokens shape: {register_tokens.shape}")
    if register_tokens.shape[1] > 0:
        print(f"   register_tokens std: {float(register_tokens.std()):.4f} (期望 ~0.02)")
    else:
        print(f"   register_tokens 为空 (num_register_tokens=0)")
    print()

def test_rope_angles_calculation():
    print("=" * 60)
    print("测试 RoPE angles 计算")
    print("=" * 60)
    
    paddle.set_device('cpu')
    model = DINOv3_vits14(class_num=10)
    
    x = paddle.randn([2, 3, 224, 224])
    
    model.eval()
    cos, sin = model.rope_embeddings(x)
    
    num_patches = (224 // 14) * (224 // 14)
    head_dim = 384 // 6
    
    print(f"✓ cos shape: {cos.shape} (期望: [{num_patches}, {head_dim}])")
    print(f"✓ sin shape: {sin.shape} (期望: [{num_patches}, {head_dim}])")
    
    assert cos.shape[0] == num_patches, f"第一维应该是 {num_patches}, 实际是 {cos.shape[0]}"
    assert cos.shape[1] == head_dim, f"第二维应该是 {head_dim}, 实际是 {cos.shape[1]}"
    
    print(f"✓ RoPE angles 维度正确！")
    print()

def test_attention_dropout():
    print("=" * 60)
    print("测试 Attention dropout")
    print("=" * 60)
    
    paddle.set_device('cpu')
    
    model = DINOv3_vits14(
        class_num=10,
        attn_drop_rate=0.1
    )
    
    assert hasattr(model.layers[0].attention, 'dropout'), "Attention 应该有 dropout 属性"
    print(f"✓ Attention dropout 值: {model.layers[0].attention.dropout}")
    
    x = paddle.randn([2, 3, 224, 224])
    
    model.train()
    out1 = model(x)
    out2 = model(x)
    
    diff = float((out1 - out2).abs().mean())
    print(f"✓ 训练模式下两次前向传播差异: {diff:.6f} (应该 > 0 因为有dropout)")
    
    model.eval()
    out3 = model(x)
    out4 = model(x)
    
    diff_eval = float((out3 - out4).abs().mean())
    print(f"✓ 评估模式下两次前向传播差异: {diff_eval:.6f} (应该 ~ 0)")
    
    assert diff > diff_eval * 10, "训练模式的差异应该明显大于评估模式"
    print(f"✓ Attention dropout 工作正常！")
    print()

def test_register_tokens():
    print("=" * 60)
    print("测试 register_tokens 处理")
    print("=" * 60)
    
    paddle.set_device('cpu')
    
    model_with_reg = DINOv3_vits14(class_num=10, num_register_tokens=4)
    model_without_reg = DINOv3_vits14(class_num=10, num_register_tokens=0)
    
    x = paddle.randn([2, 3, 224, 224])
    
    embed_with = model_with_reg.embeddings(x)
    embed_without = model_without_reg.embeddings(x)
    
    num_patches = (224 // 14) * (224 // 14)
    
    expected_with = 1 + 4 + num_patches
    expected_without = 1 + 0 + num_patches
    
    print(f"✓ 有 register_tokens 的序列长度: {embed_with.shape[1]} (期望: {expected_with})")
    print(f"✓ 无 register_tokens 的序列长度: {embed_without.shape[1]} (期望: {expected_without})")
    
    assert embed_with.shape[1] == expected_with
    assert embed_without.shape[1] == expected_without
    
    print(f"✓ register_tokens 处理正确！")
    print()

def test_end_to_end():
    print("=" * 60)
    print("端到端测试")
    print("=" * 60)
    
    paddle.set_device('cpu')
    
    configs = [
        ("vits14", dict(img_size=518, patch_size=14, embed_dim=384, depth=12, num_heads=6)),
        ("vitb14", dict(img_size=518, patch_size=14, embed_dim=768, depth=12, num_heads=12)),
    ]
    
    for name, cfg in configs:
        model = dinov3.DINOv3ViTModel(
            class_num=1000,
            **cfg,
            num_register_tokens=4,
            attn_drop_rate=0.1
        )
        
        x = paddle.randn([1, 3, cfg['img_size'], cfg['img_size']])
        
        model.eval()
        out = model(x)
        
        print(f"✓ {name}: 输入 {list(x.shape)} -> 输出 {list(out.shape)}")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DINOv3 问题修复验证测试")
    print("=" * 60 + "\n")
    
    try:
        test_embeddings_initialization()
        test_rope_angles_calculation()
        test_register_tokens()
        test_attention_dropout()
        test_end_to_end()
        
        print("=" * 60)
        print("✅ 所有测试通过！修复成功！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
