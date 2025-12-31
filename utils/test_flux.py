import torch
from diffusers import Flux2Pipeline
from huggingface_hub import get_token
import requests
import io

# 使用 4-bit 量化版本
repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:1"

# 定义远程文本编码器函数（节省本地显存）
def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"}
    )
    return torch.load(io.BytesIO(response.content)).to(device)

# 加载模型
pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    text_encoder=None, # 本地不加载巨大的文本编码器
    torch_dtype=torch.bfloat16
).to(device)

prompt = "一只寄居蟹用可乐罐当壳的写实微距摄影，沙滩，阳光，自然光影。"

# 生成图像
image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")