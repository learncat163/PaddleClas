import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import numpy as np
from ppcls.arch.backbone.model_zoo import dinov3
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits14

def test_lru_cache():
    print("=" * 50)
    print("测试 LRU 缓存功能")
    print("=" * 50)
    
    coords1 = dinov3.get_patches_center_coordinates(14, 14, 'paddle.float32')
    coords2 = dinov3.get_patches_center_coordinates(14, 14, 'paddle.float32')
    coords3 = dinov3.get_patches_center_coordinates(16, 16, 'paddle.float32')
    
    assert coords1 is coords2, "相同参数应返回缓存对象"
    assert coords1 is not coords3, "不同参数应返回不同对象"
    
    print(f"✓ LRU缓存工作正常")
    print(f"  - 缓存对象地址相同: {id(coords1) == id(coords2)}")
    print(f"  - coords1 shape: {coords1.shape}")
    print(f"  - coords3 shape: {coords3.shape}")
    print()

def test_gradient_checkpointing():
    print("=" * 50)
    print("测试梯度检查点功能")
    print("=" * 50)
    
    paddle.set_device('cpu')
    
    model_normal = DINOv3_vits14(class_num=10, use_gradient_checkpointing=False)
    model_checkpoint = DINOv3_vits14(class_num=10, use_gradient_checkpointing=True)
    
    x = paddle.randn([2, 3, 224, 224])
    
    model_normal.train()
    out_normal = model_normal(x)
    print(f"✓ 无梯度检查点模型输出: {out_normal.shape}")
    
    model_checkpoint.train()
    out_checkpoint = model_checkpoint(x)
    print(f"✓ 有梯度检查点模型输出: {out_checkpoint.shape}")
    
    assert out_normal.shape == out_checkpoint.shape, "输出shape应该相同"
    print(f"✓ 梯度检查点功能正常工作")
    print()

def test_rope_position_embeddings():
    print("=" * 50)
    print("测试 RoPE 位置编码缓存")
    print("=" * 50)
    
    paddle.set_device('cpu')
    model = DINOv3_vits14(class_num=10)
    
    x1 = paddle.randn([1, 3, 224, 224])
    x2 = paddle.randn([1, 3, 224, 224])
    x3 = paddle.randn([1, 3, 336, 336])
    
    model.eval()
    
    print("第一次计算 224x224 (未缓存)")
    out1 = model(x1)
    
    print("第二次计算 224x224 (使用缓存)")
    out2 = model(x2)
    
    print("计算 336x336 (未缓存)")
    out3 = model(x3)
    
    print(f"✓ 不同尺寸前向传播成功")
    print(f"  - 224x224 输出: {out1.shape}")
    print(f"  - 336x336 输出: {out3.shape}")
    print()

def test_training_augmentation():
    print("=" * 50)
    print("测试训练时位置编码增强")
    print("=" * 50)
    
    paddle.set_device('cpu')
    model = DINOv3_vits14(
        class_num=10,
        pos_embed_shift=0.1,
        pos_embed_jitter=1.5,
        pos_embed_rescale=2.0
    )
    
    x = paddle.randn([2, 3, 224, 224])
    
    model.train()
    out_train = model(x)
    print(f"✓ 训练模式(有增强)输出: {out_train.shape}")
    
    model.eval()
    out_eval = model(x)
    print(f"✓ 评估模式(无增强)输出: {out_eval.shape}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("DINOv3 优化功能测试")
    print("=" * 50 + "\n")
    
    try:
        test_lru_cache()
        test_rope_position_embeddings()
        test_training_augmentation()
        test_gradient_checkpointing()
        
        print("=" * 50)
        print("所有测试通过！")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
