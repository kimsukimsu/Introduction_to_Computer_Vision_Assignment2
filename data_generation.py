import os
import argparse
from PIL import Image
from tqdm import tqdm
from model import WeatherGenerator


DEFAULT_PROMPTS = {
    "rain": "make it heavy rain, wet ground, stormy weather",
    "snow": "make it snowy, snow on the ground, winter storm",
    "fog":  "add thick fog, hazy atmosphere, low visibility"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic weather images for Market-1501.")
    parser.add_argument("--source", type=str, required=True, help="Path to source Market-1501 dataset")
    parser.add_argument("--output", type=str, default="./Market_Weather", help="Base directory to save images")
    parser.add_argument("--model_id", type=str, default="timbrooks/instruct-pix2pix", help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--max_images", type=int, default=None, help="Number of images to process (default: all)")
    parser.add_argument("--weather", type=str, default="all", choices=["rain", "snow", "fog", "all"], help="Weather type")
    return parser.parse_args()

def main():
    args = parse_args()
    tasks = []
    if args.weather == "all":
        for w_type, prompt in DEFAULT_PROMPTS.items():
            tasks.append({"type": w_type, "prompt": prompt})
    else:
        tasks.append({"type": args.weather, "prompt": DEFAULT_PROMPTS[args.weather]})

    #model load
    print(f"\n>>> Loading Model: {args.model_id} on {args.device}...")
    generator = WeatherGenerator(model_id=args.model_id, device=args.device)

    #prepare images data files
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' not found.")
        return

    all_files = sorted([f for f in os.listdir(args.source) if f.lower().endswith(('.jpg', '.png'))])
    target_files = all_files[:args.max_images] if args.max_images else all_files
    
    print(f">>> Found {len(all_files)} images. Processing {len(target_files)} images.\n")

    #generate images
    for task in tasks:
        weather_type = task['type']
        prompt = task['prompt']
        
        save_dir = os.path.join(args.output, weather_type)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[*] Task: {weather_type.upper()} | Prompt: '{prompt}'")
        
        # tqdm(리스트, desc="설명글", unit="단위")
        for img_name in tqdm(target_files, desc=f"Generating {weather_type}", unit="img"):
            img_path = os.path.join(args.source, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB') #open images
                result_image = generator.generate(image, prompt) #generate image 
                
                save_name = f"{os.path.splitext(img_name)[0]}_{weather_type}.jpg"
                result_image.save(os.path.join(save_dir, save_name)) #save
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        
        print(f"Generated images saved to {save_dir}\n")

if __name__ == "__main__":
    main()