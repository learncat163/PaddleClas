# DINOv3 TIPC 集成说明

## 概述

将 DINOv3 的精度对比测试集成到 PaddleClas 的 TIPC (Training and Inference Pipeline Certification) 测试框架中。

## 目录结构

```
test_tipc/
├── configs/
│   └── DINOv3/
│       └── DINOv3_vits16_train_infer_python.txt    # TIPC 配置文件
├── compare_hf_precision.py                          # HF vs Paddle 精度对比脚本
├── test_dinov3_hf_precision.sh                      # DINOv3 精度测试入口脚本
└── ...
```

## 使用方法

### 1. 基础精度对比测试

```bash
# 测试 DINOv3_vits16 模型精度
bash test_tipc/test_dinov3_hf_precision.sh \
    test_tipc/configs/DINOv3/DINOv3_vits16_train_infer_python.txt \
    lite_train_lite_infer
```

### 2. 直接使用 Python 脚本

```bash
# 直接调用精度对比脚本
python test_tipc/compare_hf_precision.py \
    --model_name DINOv3_vits16 \
    --hf_model_path /path/to/hf/model \
    --pdparams_path /path/to/pdparams \
    --image_path /path/to/test/image.jpg \
    --atol 1e-5 \
    --rtol 1e-5
```

### 3. 集成到现有 TIPC 流程

在现有的 `test_train_inference_python.sh` 中添加精度验证步骤：

```bash
# 在推理完成后添加 HF 精度验证
if [ "${model_name}" == "DINOv3"* ]; then
    bash test_tipc/test_dinov3_hf_precision.sh ${FILENAME} ${MODE}
fi
```

## 配置文件说明

`test_tipc/configs/DINOv3/DINOv3_vits16_train_infer_python.txt` 新增字段：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `hf_model_path` | HuggingFace 模型路径 | - |
| `hf_precision_atol` | 绝对误差容忍度 | 1e-5 |
| `hf_precision_rtol` | 相对误差容忍度 | 1e-5 |
| `test_image_path` | 测试图片路径 | (预设路径) |

## 输出结果

测试完成后会在 `test_tipc/output/${model_name}/${MODE}/` 目录生成：

- `hf_precision.log` - 精度测试日志
- JSON 格式的测试结果

示例结果：
```json
{
  "max_diff": 8.5e-07,
  "mean_diff": 2.1e-07,
  "paddle_shape": [1, 384],
  "torch_shape": [1, 384],
  "passed": true,
  "status": "success",
  "model_name": "DINOv3_vits16"
}
```

## 扩展其他模型

要为其他 DINOv3 变体添加测试：

1. 创建对应的配置文件：`DINOv3_vitb16_train_infer_python.txt`
2. 更新模型名称和权重路径
3. `test_dinov3_hf_precision.sh` 会自动匹配对应的模型类
