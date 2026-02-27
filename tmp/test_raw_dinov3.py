import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name,
    device_map="auto",
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
print("数据类型:", pooled_output.dtype)
print("前4个特征值:", pooled_output[0, :4].tolist())