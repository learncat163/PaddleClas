import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits14

def example_classification():
    """示例1: 图像分类"""
    print("\n示例1: 图像分类")
    print("-" * 50)
    
    model = DINOv3_vits14(class_num=1000)
    model.eval()
    
    x = paddle.randn([1, 3, 224, 224])
    
    with paddle.no_grad():
        logits = model(x)
        probs = paddle.nn.functional.softmax(logits, axis=-1)
        top5_idx = paddle.topk(probs, k=5)[1]
    
    print(f"输入: {list(x.shape)}")
    print(f"输出 logits: {list(logits.shape)}")
    print(f"Top-5 预测类别: {top5_idx.numpy()[0]}")


def example_feature_extraction():
    """示例2: 特征提取（用于相似度检索）"""
    print("\n示例2: 特征提取（用于相似度检索）")
    print("-" * 50)
    
    model = DINOv3_vits14(class_num=0)
    model.eval()
    
    images = paddle.randn([8, 3, 224, 224])
    
    with paddle.no_grad():
        features = model.forward_features(images)
        
        features_norm = paddle.nn.functional.normalize(features, axis=-1)
        similarity = paddle.matmul(features_norm, features_norm.T)
    
    print(f"输入批量: {list(images.shape)}")
    print(f"特征向量: {list(features.shape)}")
    print(f"相似度矩阵: {list(similarity.shape)}")
    print(f"图片0与图片1的相似度: {float(similarity[0, 1]):.4f}")


def example_backbone():
    """示例3: 作为 Backbone 提取多层特征"""
    print("\n示例3: 作为 Backbone 提取多层特征")
    print("-" * 50)
    
    model = DINOv3_vits14(
        class_num=0,
        out_indices=[2, 5, 8, 11]
    )
    model.eval()
    
    x = paddle.randn([1, 3, 518, 518])
    
    with paddle.no_grad():
        multi_level_features = model(x)
    
    print(f"输入: {list(x.shape)}")
    print(f"输出层数: {len(multi_level_features)}")
    for i, feat in enumerate(multi_level_features):
        print(f"  第 {i} 层特征: {list(feat.shape)}")


def example_multi_scale():
    """示例4: 多尺度推理"""
    print("\n示例4: 多尺度推理")
    print("-" * 50)
    
    model = DINOv3_vits14(class_num=1000)
    model.eval()
    
    scales = [224, 336, 518]
    
    for scale in scales:
        x = paddle.randn([1, 3, scale, scale])
        with paddle.no_grad():
            output = model(x)
        print(f"尺寸 {scale}x{scale}: 输入{list(x.shape)} -> 输出{list(output.shape)}")


def example_batch_inference():
    """示例5: 批量推理"""
    print("\n示例5: 批量推理")
    print("-" * 50)
    
    model = DINOv3_vits14(class_num=1000)
    model.eval()
    
    batch_data = paddle.randn([16, 3, 224, 224])
    
    with paddle.no_grad():
        predictions = model(batch_data)
        probs = paddle.nn.functional.softmax(predictions, axis=-1)
        pred_classes = paddle.argmax(probs, axis=-1)
    
    print(f"批量输入: {list(batch_data.shape)}")
    print(f"预测结果: {list(pred_classes.shape)}")
    print(f"前5个样本的预测类别: {pred_classes[:5].numpy()}")


if __name__ == "__main__":
    paddle.set_device('cpu')
    
    print("=" * 60)
    print("DINOv3 推理实战示例")
    print("=" * 60)
    
    example_classification()
    example_feature_extraction()
    example_backbone()
    example_multi_scale()
    example_batch_inference()
    
    print("\n" + "=" * 60)
    print("更多用法:")
    print("=" * 60)
    print("""
# 1. 使用不同的模型变体
from ppcls.arch.backbone.model_zoo.dinov3 import (
    DINOv3_vits14,  # 小模型，384维
    DINOv3_vitb14,  # 中等模型，768维
    DINOv3_vitg14,  # 大模型，1536维
)

# 2. 加载预训练权重
model = DINOv3_vitb14(class_num=1000)
# model.load_hf_pretrained('facebook/dinov2-base')

# 3. 使用梯度检查点（训练时节省显存）
model_train = DINOv3_vits14(
    class_num=1000,
    use_gradient_checkpointing=True
)

# 4. 使用 register tokens（与官方模型一致）
model_reg = DINOv3_vits14(
    class_num=1000,
    num_register_tokens=4
)

# 5. 自定义训练配置
model_custom = DINOv3_vits14(
    class_num=100,          # 分类数
    drop_path_rate=0.1,     # DropPath 比例
    attn_drop_rate=0.1,     # Attention Dropout
    pos_embed_shift=0.1,    # 位置编码增强
    pos_embed_jitter=1.5,
    pos_embed_rescale=2.0,
)
""")
    
    print("\n完成！")
