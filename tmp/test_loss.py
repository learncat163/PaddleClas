"""
精度测试脚本，需要同时安装paddleclas和 torch 和 transformers，测试环境如下：
1. 最新的 paddleclas 的develop分支 + 当前的PR合并请求
2. paddlepaddle-gpu==3.3.0
3. numpy==1.24.4
4. torch==2.9.0+cu130
5. torchvision==0.24.0+cu130
6. transformers==5.2.0
"""
import os
import numpy as np
import paddle
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from ppcls.arch.backbone.model_zoo.dinov3 import DINOv3_vits16, DINOv3_vitb16, DINOv3_vitl16


class ModelConfig:
    def __init__(self, name, pd_class, pd_weights, hf_path, dim):
        self.name = name
        self.pd_class = pd_class
        self.pd_weights = pd_weights
        self.hf_path = hf_path
        self.dim = dim

# dinov3的pd版本的模型在：  https://aistudio.baidu.com/modelsdetail/44390?modelId=44390，或者
# https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/dinov3-vitb16.pdparams
# https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/dinov3-vitl16.pdparams
# https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/dinov3-vits16.pdparams
# HF原始模型地址参考：https://aistudio.baidu.com/modelsdetail/44390?modelId=44390 的说明

diff_models_config = [
    ModelConfig("DINOv3_vits16", DINOv3_vits16, "/tmp/dinov3-vits16.pdparams",
                os.path.expanduser("~/llm/facebook/dinov3-vits16-pretrain-lvd1689m/"), 384),
    ModelConfig("DINOv3_vitb16", DINOv3_vitb16, "/tmp/dinov3-vitb16.pdparams",
                os.path.expanduser("~/llm/facebook/dinov3-vitb16-pretrain-lvd1689m/"), 768),
    ModelConfig("DINOv3_vitl16", DINOv3_vitl16, "/tmp/dinov3-vitl16.pdparams",
                os.path.expanduser("~/llm/facebook/dinov3-vitl16-pretrain-lvd1689m/"), 1024),
]


image_path = "docs/images/inference_deployment/whl_demo.jpg"
image = Image.open(image_path).convert('RGB')

hf_model_path = os.path.expanduser("~/llm/facebook/dinov3-vits16-pretrain-lvd1689m/")
hf_processor = AutoImageProcessor.from_pretrained(hf_model_path)
inputs_hf_image = hf_processor(images=image, return_tensors="pt")
pixel_values_hf = inputs_hf_image['pixel_values']

image_tensor = TF.to_tensor(image)
resized = TF.resize(image_tensor, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
image_array = resized.numpy()
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
image_array = (image_array - mean) / std
image_array = np.expand_dims(image_array, axis=0)
pixel_values_pd = paddle.to_tensor(image_array, dtype='float32')



print("="*70)
print("DINOv3 Paddle vs Torch 精度对比")
print("="*70)

for cfg in diff_models_config:
    print(f"\n{'='*70}")
    print(f"模型: {cfg.name}")
    print(f"{'='*70}")

    hf_model = AutoModel.from_pretrained(cfg.hf_path)
    hf_model.eval()
    with torch.inference_mode():
        hf_outputs = hf_model(pixel_values_hf)
        hf_pooled = hf_outputs.pooler_output

    pd_model = cfg.pd_class(class_num=0, pretrained=False)
    pd_model.eval()
    state_dict = paddle.load(cfg.pd_weights)
    pd_model.set_state_dict(state_dict)
    with paddle.no_grad():
        pd_pooled = pd_model.forward_features(pixel_values_pd)

    hf_np = hf_pooled.detach().cpu().numpy()
    pd_np = pd_pooled.numpy()

    abs_diff = np.abs(hf_np - pd_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"输出 shape: {hf_np.shape}")
    print(f"整体最大误差: {max_diff:.2e}")
    print(f"整体平均误差: {mean_diff:.2e}")
    print(f"HF 前4个值: {hf_np[0, :4].tolist()}")
    print(f"PD 前4个值: {pd_np[0, :4].tolist()}")
    print(f"状态: {'[PASS]' if max_diff < 1e-5 else '[FAIL]'}")

print(f"\n{'='*70}")
print("测试完成")
print(f"{'='*70}")
