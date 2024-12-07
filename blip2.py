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

def generate_caption(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    img = Image.open(image_path).convert("RGB")
    prompt = "Describe the aesthetic and style of this image, focusing primarily on its fashion vibe and overall mood. Only mention key fashion elements, such as colors, textures, or notable design features and their vibe/mood, but don't describe the person wearing it."
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=150)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption

# Dictionary to store captions: { "product_id" or "scene_id" : "caption" }
captions = {}

subset = 10
with tqdm(total=len(fashion_data[:subset]), desc="Generating captions") as pbar:
    for item in fashion_data[:subset]:
        product_id = item["product"]
        scene_id = item["scene"]
        print("Product ID:", product_id)
        print("Scene ID:", scene_id)
        product_path = os.path.join("images", "product", f"{product_id}.jpg")
        scene_path = os.path.join("images", "scene", f"{scene_id}.jpg")

        # Generate product caption if not already done
        if product_id not in captions:
            product_caption = generate_caption(product_path)
            print("Product caption:", product_caption)
            if product_caption is not None:
                captions[product_id] = product_caption

        # Generate scene caption if not already done
        if scene_id not in captions:
            scene_caption = generate_caption(scene_path)
            print("Scene caption:", scene_caption)
            if scene_caption is not None:
                captions[scene_id] = scene_caption

        pbar.update(1)

# Save captions to a JSON file
with open("captions.json", "w") as f:
    json.dump(captions, f, indent=2)

print("Captions saved to captions.json")
