import os
import kagglehub
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import json
from tqdm import tqdm

print(torch.backends.mps.is_available())

# Download the dataset (if not already present)
path = kagglehub.dataset_download("pypiahmad/shop-the-look-dataset")
fashion_data = []
with open(os.path.join(path, 'fashion.json'), 'r') as f:
    for line in f:
        try:
            fashion_data.append(json.loads(line.strip()))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# Load a BLIP-2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/instructblip-vicuna-13b"
processor = InstructBlipProcessor.from_pretrained(model_name)
model = InstructBlipForConditionalGeneration.from_pretrained(model_name).to(device)

def generate_captions_batch(image_paths):
    # Filter out paths of non-existent images
    valid_paths = [path for path in image_paths if os.path.exists(path)]
    if not valid_paths:
        print("No valid images found.")
        return []

    images = [Image.open(path).convert("RGB") for path in valid_paths]
    prompts = ["Describe the fashion aesthetic of this image in a moody, chic tone. Focus on the colors, textures, and notable design elements, evoking the emotion or mood it represents. Suggest an occasion or situation where this style would shine. Avoid describing the person wearing it and concentrate on the outfit or scene's essence and vibe."] * len(images)
    
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=150)
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Strip the prompt part for each caption
    prompt_length = len(prompts[0].split())
    cleaned_captions = [" ".join(caption.split()[prompt_length:]).strip() for caption in captions]
    return cleaned_captions


# Dictionary to store captions: { "product_id" or "scene_id" : "caption" }
captions = {}

subset = 3
with tqdm(total=len(fashion_data[:subset]), desc="Generating captions") as pbar:
    for item in fashion_data[:subset]:
        product_id = item["product"]
        scene_id = item["scene"]
        print("Product ID:", product_id)
        print("Scene ID:", scene_id)
        product_path = os.path.join("images", "product", f"{product_id}.jpg")
        scene_path = os.path.join("images", "scene", f"{scene_id}.jpg")

        if product_id not in captions:
            product_caption = generate_captions_batch(product_path)
            print("Product caption:", product_caption)
            if product_caption is not None:
                captions[product_id] = product_caption

        if scene_id not in captions:
            scene_caption = generate_captions_batch(scene_path)
            print("Scene caption:", scene_caption)
            if scene_caption is not None:
                captions[scene_id] = scene_caption

        pbar.update(1)

# Save captions to a JSON file
with open("captions.json", "w") as f:
    json.dump(captions, f, indent=2)

print("Captions saved to captions.json")
