import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "/home/cao/code/self/paddle/dinov3/000000039769.jpg"
image = load_image(url)


def print_model_output(model_path, input_image, print_banner):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
    )

    inputs = processor(images=input_image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    print(print_banner + " test infer:")
    print("first 4 outputs:", pooled_output[0, :4].tolist())


print_model_output("/home/cao/llm/facebook/dinov3-vits16-pretrain-lvd1689m/", image, "vits16")

print_model_output("/home/cao/llm/facebook/dinov3-vitb16-pretrain-lvd1689m/", image, "vitb16")


print_model_output("/home/cao/llm/facebook/dinov3-vitl16-pretrain-lvd1689m/", image, "vitl16")