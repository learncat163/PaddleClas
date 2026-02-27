import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits14, DINOv3_vitb14

print("=" * 70)
print("DINOv3 推理使用指南")
print("=" * 70)

print("\n" + "=" * 70)
print("1. 图像分类任务（默认模式）")
print("=" * 70)

paddle.set_device('cpu')
model = DINOv3_vits14(class_num=1000)
model.eval()

x = paddle.randn([1, 3, 224, 224])
output = model(x)

print(f"输入: {list(x.shape)}")
print(f"输出: {list(output.shape)}")
print(f"说明: 输出是分类 logits，shape [batch_size, num_classes]")

print("\n" + "=" * 70)
print("2. 特征提取（Backbone 模式）")
print("=" * 70)

model_backbone = DINOv3_vits14(
    class_num=0,
    out_indices=[3, 5, 8, 11]
)
model_backbone.eval()

x = paddle.randn([2, 3, 518, 518])
features = model_backbone(x)

print(f"输入: {list(x.shape)}")
print(f"输出特征数: {len(features)}")
for i, feat in enumerate(features):
    print(f"  特征 {i}: shape={list(feat.shape)}")
print(f"说明: 输出多层特征，用于下游任务（检测、分割等）")

print("\n" + "=" * 70)
print("3. 获取 [CLS] token 特征（用于检索、聚类）")
print("=" * 70)

model_feat = DINOv3_vits14(class_num=0)
model_feat.eval()

x = paddle.randn([4, 3, 224, 224])
pooled_feature = model_feat.forward_features(x)

print(f"输入: {list(x.shape)}")
print(f"输出: {list(pooled_feature.shape)}")
print(f"说明: 输出 [CLS] token 的特征向量，可用于相似度计算")

print("\n" + "=" * 70)
print("4. 加载 HuggingFace 预训练权重")
print("=" * 70)

print("""
model = DINOv3_vitb14(class_num=1000)
model.load_hf_pretrained('facebook/dinov2-base')

x = paddle.randn([1, 3, 224, 224])
model.eval()
output = model(x)

说明: 自动从 HuggingFace Hub 下载并转换权重
""")

print("\n" + "=" * 70)
print("5. 不同尺寸的推理")
print("=" * 70)

model_multi = DINOv3_vits14(class_num=1000)
model_multi.eval()

sizes = [224, 336, 518]
for size in sizes:
    x = paddle.randn([1, 3, size, size])
    output = model_multi(x)
    print(f"输入尺寸: {size}x{size} -> 输出: {list(output.shape)}")

print(f"\n说明: DINOv3 支持任意尺寸输入（需要是 patch_size 的倍数）")

print("\n" + "=" * 70)
print("6. 批量推理示例")
print("=" * 70)

model_batch = DINOv3_vits14(class_num=1000)
model_batch.eval()

batch_size = 8
x = paddle.randn([batch_size, 3, 224, 224])

with paddle.no_grad():
    output = model_batch(x)
    probs = paddle.nn.functional.softmax(output, axis=-1)
    top5_idx = paddle.topk(probs, k=5, axis=-1)[1]

print(f"批量大小: {batch_size}")
print(f"输出概率: {list(probs.shape)}")
print(f"Top-5 类别索引: {list(top5_idx.shape)}")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
DINOv3 在 PaddleClas 中的使用：

1. 主要类: DINOv3ViTModel
   - 通过工厂函数使用: DINOv3_vits14, DINOv3_vitb14, DINOv3_vitg14 等

2. 使用场景:
   - 图像分类: class_num > 0
   - 特征提取: class_num = 0
   - Backbone: 设置 out_indices 参数

3. 关键参数:
   - class_num: 分类类别数，0 表示不使用分类头
   - out_indices: 输出哪些层的特征 (Backbone 模式)
   - num_register_tokens: register tokens 数量（默认0，官方模型用4）
   - use_gradient_checkpointing: 是否使用梯度检查点（节省显存）

4. 支持特性:
   ✓ 多尺度输入
   ✓ 批量推理
   ✓ HuggingFace 权重加载
   ✓ 梯度检查点
   ✓ RoPE 位置编码缓存优化
""")

print("\n完整示例代码已生成！")
