# DINOv3 自动化训练、测试、推理配置任务规划

## 项目背景

1. **DINOv3 已完成迁移**：运行误差在 1e-6 之内，现有迁移代码不需修改
2. **预训练权重位置**：`/tmp/` 目录下
   - `dinov3-vits16.pdparams` - Small 模型
   - `dinov3-vitb16.pdparams` - Base 模型
   - `dinov3-vitl16.pdparams` - Large 模型

## 任务目标

为 DINOv3 的三个不同规模模型创建完整的自动化配置和脚本体系。

---

## 文件结构规划

```
PaddleClas/
├── ppcls/configs/ImageNet/DINOv3/          # 新建：YAML 训练配置
│   ├── DINOv3_vits16_patch16_224.yaml     # Small 模型配置
│   ├── DINOv3_vitb16_patch16_224.yaml     # Base 模型配置
│   └── DINOv3_vitl16_patch16_224.yaml     # Large 模型配置
│
└── test_tipc/configs/DINOv3/                # 已存在，需补充
    ├── DINOv3_vits16_train_infer_python.txt
    ├── DINOv3_vitb16_train_infer_python.txt
    ├── DINOv3_vitl16_train_infer_python.txt
    ├── DINOv3_vits16_hf_precision.txt
    ├── DINOv3_vitb16_hf_precision.txt
    └── DINOv3_vitl16_hf_precision.txt

**注意**：推理配置使用现有的 `deploy/configs/inference_cls.yaml`，不需要单独创建
```

---

## 实施步骤

### 阶段 1：创建 YAML 训练配置文件

#### 1.1 DINOv3_vits16_patch16_224.yaml
- 路径：`ppcls/configs/ImageNet/DINOv3/DINOv3_vits16_patch16_224.yaml`
- 参数：
  - 模型：ViT-S/16, 384 维输出
  - 图像尺寸：224x224
  - 预训练权重：`/tmp/dinov3-vits16.pdparams`

#### 1.2 DINOv3_vitb16_patch16_224.yaml
- 路径：`ppcls/configs/ImageNet/DINOv3/DINOv3_vitb16_patch16_224.yaml`
- 参数：
  - 模型：ViT-B/16, 768 维输出
  - 图像尺寸：224x224
  - 预训练权重：`/tmp/dinov3-vitb16.pdparams`

#### 1.3 DINOv3_vitl16_patch16_224.yaml
- 路径：`ppcls/configs/ImageNet/DINOv3/DINOv3_vitl16_patch16_224.yaml`
- 参数：
  - 模型：ViT-L/16, 1024 维输出
  - 图像尺寸：224x224
  - 预训练权重：`/tmp/dinov3-vitl16.pdparams`

**执行命令**：
```bash
# 创建目录
mkdir -p /home/cao/code/github/PaddleClas/ppcls/configs/ImageNet/DINOv3
```

---

### 阶段 2：完善 TIPC 测试配置

#### 2.1 DINOv3_vitb16_train_infer_python.txt
- 基于现有 vits16 配置，扩展到 vitb16

#### 3.2 DINOv3_vitl16_train_infer_python.txt
- 基于现有 vits16 配置，扩展到 vitl16

#### 3.3 更新 HF 精度对比配置
- DINOv3_vitb16_hf_precision.txt
- DINOv3_vitl16_hf_precision.txt

**执行命令**：
```bash
# 目录已存在，直接添加文件
cd /home/cao/code/github/PaddleClas/test_tipc/configs/DINOv3/
```

---

### 阶段 3：测试脚本

#### 3.1 训练测试（用于 fine-tune 场景）
```bash
# Small 模型训练测试
python tools/train.py -c ppcls/configs/ImageNet/DINOv3/DINOv3_vits16_patch16_224.yaml \
  -o Global.epochs=1 \
  -o Global.output_dir=./output/dinov3_vits16_test \
  -o Arch.pretrained=/tmp/dinov3-vits16.pdparams
```

#### 3.2 推理测试
```bash
# 导出推理模型
python tools/export_model.py -c ppcls/configs/ImageNet/DINOv3/DINOv3_vits16_patch16_224.yaml \
  -o Global.pretrained_model=/tmp/dinov3-vits16.pdparams \
  -o Global.save_inference_dir=./inference/dinov3_vits16

# 推理测试（使用标准 inference_cls.yaml）
python deploy/python/predict_cls.py \
  -c deploy/configs/inference_cls.yaml \
  -o Global.inference_model_dir=./inference/dinov3_vits16 \
  -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg
```

#### 3.3 TIPC 测试
```bash
cd test_tipc
bash test_train_inference_python.sh test_tipc/configs/DINOv3/DINOv3_vits16_train_infer_python.txt
```

---

## 测试命令记录

### 1. 快速推理验证（不需要训练数据）

```bash
# 使用预训练权重直接推理
cd /home/cao/code/github/PaddleClas

# Small 模型
python tools/infer.py \
  -c ppcls/configs/ImageNet/DINOv3/DINOv3_vits16_patch16_224.yaml \
  -o Global.pretrained_model=/tmp/dinov3-vits16.pdparams \
  -o Infer.infer_imgs=docs/images/inference_deployment/whl_demo.jpg

# Base 模型
python tools/infer.py \
  -c ppcls/configs/ImageNet/DINOv3/DINOv3_vitb16_patch16_224.yaml \
  -o Global.pretrained_model=/tmp/dinov3-vitb16.pdparams \
  -o Infer.infer_imgs=docs/images/inference_deployment/whl_demo.jpg

# Large 模型
python tools/infer.py \
  -c ppcls/configs/ImageNet/DINOv3/DINOv3_vitl16_patch16_224.yaml \
  -o Global.pretrained_model=/tmp/dinov3-vitl16.pdparams \
  -o Infer.infer_imgs=docs/images/inference_deployment/whl_demo.jpg
```

### 2. 导出推理模型

```bash
# Small 模型导出
python tools/export_model.py \
  -c ppcls/configs/ImageNet/DINOv3/DINOv3_vits16_patch16_224.yaml \
  -o Global.pretrained_model=/tmp/dinov3-vits16.pdparams \
  -o Global.save_inference_dir=./inference/dinov3_vits16
```

### 3. 使用导出模型推理

```bash
python deploy/python/predict_cls.py \
  -c deploy/configs/dinov3/inference_dinov3_vits16.yaml \
  -o Global.inference_model_dir=./inference/dinov3_vits16 \
  -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg
```

---

## 任务进度

- [x] 阶段 1：创建 YAML 训练配置文件（3个）✅
- [x] 阶段 2：完善 TIPC 测试配置（补充 4 个文件）✅
- [ ] 阶段 3：运行测试验证

---

## 已创建文件清单

### 阶段 1：训练配置文件（ppcls/configs/ImageNet/DINOv3/）
- [x] `DINOv3_vits16_patch16_224.yaml` - Small 模型配置
- [x] `DINOv3_vitb16_patch16_224.yaml` - Base 模型配置
- [x] `DINOv3_vitl16_patch16_224.yaml` - Large 模型配置

### 阶段 2：TIPC 测试配置（test_tipc/configs/DINOv3/）
- [x] `DINOv3_vits16_train_infer_python.txt` (已更新)
- [x] `DINOv3_vitb16_train_infer_python.txt` (新建)
- [x] `DINOv3_vitl16_train_infer_python.txt` (新建)
- [x] `DINOv3_vits16_hf_precision.txt` (已存在)
- [x] `DINOv3_vitb16_hf_precision.txt` (新建)
- [x] `DINOv3_vitl16_hf_precision.txt` (新建)

---

## 注意事项

1. **DINOv3 的特性**：
   - 主要作为预训练特征提取器使用
   - 不需要从头训练（需要 1.7B 图像）
   - 可以基于预训练权重进行 fine-tune

2. **配置文件要点**：
   - `Arch.name` 必须与模型类名一致
   - `Infer.transforms` 的预处理必须与原始实现一致
   - 图像归一化使用 ImageNet 标准：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

3. **权重文件路径**：
   - 当前使用 `/tmp/` 下的本地权重
   - 后续可考虑发布到模型仓库
