
import os
import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self, lora_paths):
        # lora_paths 是一个字典，将角色名称映射到LoRa路径
        # <path/to/lora.safetensors>是将C站的LoRA下载下来的*.safetensors放在本地硬盘的具体路径和文件名；
        self.lora_paths = lora_paths

    def generate_image(self, character_name, prompt, negative_prompt, width=512, height=768, guidance_scale=9, num_inference_steps=30):
        lora_path = self.lora_paths.get(character_name)
        if lora_path is None:
            raise ValueError(f"No LoRa path found for character: {character_name}")

        model = StableDiffusionPipeline.from_pretrained(
            "emilianJR/chilloutmix_NiPrunedFp32Fix",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
        model.load_lora_weights(lora_path)

        seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        image = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        output_path = f"/tmp/out-{seed}.png"
        image.save(output_path)
        return output_path

if __name__ == "__main__":
    # 使用示例
    lora_paths = {
        "Chisato": "<path/to/chisato/lora.safetensors>",
        "Fujiwara": "<path/to/fujiwara/lora.safetensors>",
        # 其他角色...
    }
    image_generator = ImageGenerator(lora_paths)

    # 生成图像
    character_name = "Chisato"
    prompt = "相应的描述"
    negative_prompt = "相应的负面描述"
    output_path = image_generator.generate_image(character_name, prompt, negative_prompt)
    print("Image saved to:", output_path)