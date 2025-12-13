import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class DataGenerator:
    def __init__(self, model_id="timbrooks/instruct-pix2pix", device="cuda"):
        # 파이프라인 로드
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            safety_checker=None
        ).to(device)
        
        # 스케줄러 설정 (오타 수정: self.pipe -> self.pipeline)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        self.device = device

    def generate(self, image, prompt):
        original_size = image.size
        
        # [수정 포인트] MVTec AD는 정사각형에 가까우므로 512x512 추천
        # 256x512는 이미지를 왜곡시킵니다.
        process_resolution = (512, 512) 
        input_image = image.resize(process_resolution) 
        
        result = self.pipeline(
            prompt, 
            image=input_image, 
            num_inference_steps=20, 
            
            # [핵심 파라미터 튜닝]
            # image_guidance_scale: 원본 이미지를 얼마나 유지할지 (높을수록 구조 유지)
            # 1.5 ~ 2.0 사이 추천. 1.2는 모양이 변할 수 있음.
            image_guidance_scale=1.5, 
            
            # guidance_scale: 프롬프트를 얼마나 따를지
            guidance_scale=7.5
        ).images[0]
        
        # 다시 원본 크기로 복원 (마스크와 매칭하기 위해)
        return result.resize(original_size)