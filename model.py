import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class WeatherGenerator:
    def __init__(self, model_id="timbrooks/instruct-pix2pix", device="cuda"):
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.device = device

    def generate(self, image, prompt):
        
        original_size = image.size
        input_image = image.resize((256, 512)) #
        
        result = self.pipeline(
            prompt, 
            image=input_image, 
            num_inference_steps=20, 
            image_guidance_scale=1.2,
            guidance_scale=7.5
        ).images[0]
        
        return result.resize(original_size)