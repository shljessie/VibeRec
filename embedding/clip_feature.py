import os
import kagglehub
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
from tqdm import tqdm

print(torch.backends.mps.is_available())

# We assume all images are already downloaded into "images/product" and "images/scene"
# directories. Each image is named "<signature>.jpg".

path = kagglehub.dataset_download("pypiahmad/shop-the-look-dataset")

fashion_data = []
with open(os.path.join(path, 'fashion.json'), 'r') as f:
    for line in f:
        try:
            fashion_data.append(json.loads(line.strip()))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def extract_clip_features(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize embeddings
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.view(1, -1)

def get_style_embedding(image_path):
    features = extract_clip_features(image_path)
    return features

embeddings = []
with tqdm(total=len(fashion_data), desc="Processing subset of fashion data") as pbar:
    for item in fashion_data:
        product_id = item["product"]
        scene_id = item["scene"]

        product_path = os.path.join("images", "product", f"{product_id}.jpg")
        scene_path = os.path.join("images", "scene", f"{scene_id}.jpg")

        product_embedding = get_style_embedding(product_path)
        scene_embedding = get_style_embedding(scene_path)

        if product_embedding is not None and scene_embedding is not None:
            embeddings.append({
                "product_id": product_id,
                "scene_id": scene_id,
                "product_embedding": product_embedding.squeeze().tolist(),
                "scene_embedding": scene_embedding.squeeze().tolist()
            })
        pbar.update(1)

with open("embeddings.json", "w") as f:
    json.dump(embeddings, f)
torch.save(embeddings, "clip_embeddings.pt")
