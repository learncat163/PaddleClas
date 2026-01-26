import paddle

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 初始化logger
from ppcls.utils import logger
logger.init_logger()

# 这里的模型可能还没合并到 paddleclas主线，需要等待合并
from ppcls.arch.backbone.model_zoo.efficientvit import (
    efficientvit_b0, efficientvit_b1, efficientvit_b2, efficientvit_b3,
    efficientvit_l1, efficientvit_l2, efficientvit_l3)

from paddle.static import InputSpec
import re
import os


def create_static_model(model_dir, model_filename, save_dir):
    # 根据 model_name 选择对应的模型类
    if model_filename.startswith("efficientvit_b0"):
        model_class = efficientvit_b0
    elif model_filename.startswith("efficientvit_b1"):
        model_class = efficientvit_b1
    elif model_filename.startswith("efficientvit_b2"):
        model_class = efficientvit_b2
    elif model_filename.startswith("efficientvit_b3"):
        model_class = efficientvit_b3
    elif model_filename.startswith("efficientvit_l1"):
        model_class = efficientvit_l1
    elif model_filename.startswith("efficientvit_l2"):
        model_class = efficientvit_l2
    elif model_filename.startswith("efficientvit_l3"):
        model_class = efficientvit_l3
    else:
        raise ValueError(f"找不到合适的模型类来处理 {model_filename}")

    # 从 model_name 中提取分辨率信息
    resolution_match = re.search(r'r(\d+)_in1k', model_filename)
    if resolution_match:
        resolution = int(resolution_match.group(1))
    else:
        raise ValueError(f"无法从模型名称 {model_filename} 中提取分辨率信息，期望格式如 r224_in1k、r256_in1k 等")

    pretrained_path = os.path.join(model_dir, model_filename)
    model = model_class(pretrained=pretrained_path, num_classes=1000)
    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[InputSpec([None, 3, resolution, resolution], dtype="float32")]
    )

    save_dir = os.path.join(save_dir, model_filename.replace(".pdparams", ""))
    os.makedirs(save_dir, exist_ok=True)
    # 默认情况下，paddleclas读取的模型 文件名是 inference.pdmodel inference.pdiparams
    paddle.jit.save(model, os.path.join(save_dir, "inference"))




def batch_convert_models():
    model_dir = "weights"
    save_dir = "/tmp/paddle_dyn_to_static"

    files = os.listdir(model_dir)

    model_files = [f for f in files if f.endswith('.pdparams') and
                   f.startswith(('efficientvit_b', 'efficientvit_l'))]

    print(f"找到 {len(model_files)} 个模型文件待转换")

    for model_name in model_files:
        print(f"正在处理: {model_name}")

        try:
            create_static_model(
                model_dir=model_dir,
                model_filename=model_name,
                save_dir=save_dir
            )
            print(f"✓ 成功转换: {model_name}")
        except Exception as e:
            print(f"✗ 转换失败: {model_name}, 错误: {str(e)}")

    print("批量转换完成")

batch_convert_models()