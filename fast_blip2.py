import os
import kagglehub
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
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

# Load a smaller BLIP-2 model for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/blip2-flan-t5-base"
processor = InstructBlipProcessor.from_pretrained(model_name, token="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")
model = InstructBlipForConditionalGeneration.from_pretrained(model_name).half().to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset for images
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            return path, transform(img)
        return None

# Function to generate captions in batches
def generate_captions_batch(image_batch, image_paths):
    prompts = ["Describe the fashion aesthetic of this image in a moody, chic tone. "
               "Focus on the colors, textures, and notable design elements, evoking the emotion or mood it represents. "
               "Suggest an occasion or situation where this style would shine. "
               "Avoid describing the person wearing it and concentrate on the outfit or scene's essence and vibe."] * len(image_batch)

    inputs = processor(images=image_batch, text=prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=150)

    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Strip the prompt text
    prompt_length = len(prompts[0].split())
    cleaned_captions = {os.path.splitext(os.path.basename(image_paths[i]))[0]: 
                         " ".join(captions[i].split()[prompt_length:]).strip()
                         for i in range(len(image_paths))}
    return cleaned_captions

# Prepare image paths
product_paths = [os.path.join("images", "product", f"{item['product']}.jpg") for item in fashion_data]
scene_paths = [os.path.join("images", "scene", f"{item['scene']}.jpg") for item in fashion_data]
all_image_paths = product_paths + scene_paths

# Create the dataset and dataloader
dataset = ImageDataset(all_image_paths)
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=lambda x: list(filter(None, x)))

# Generate captions and save them
captions = {}

with tqdm(total=len(all_image_paths), desc="Generating captions") as pbar:
    for batch in dataloader:
        image_paths, image_batch = zip(*batch)
        captions.update(generate_captions_batch(image_batch, image_paths))
        pbar.update(len(image_batch))

# Save captions to a JSON file
output_file = "captions.json"
with open(output_file, "w") as f:
    json.dump(captions, f, indent=2)

print(f"Captions saved to {output_file}")
