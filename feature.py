import requests
import os
import kagglehub
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
from tqdm import tqdm

print(torch.backends.mps.is_available())

def download_image(url, category, base_dir="images"):
    save_dir = os.path.join(base_dir, category)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, url.split("/")[-1])
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return save_path
    else:
        print(f"Failed to download image: {url}")
        return None


path = kagglehub.dataset_download("pypiahmad/shop-the-look-dataset")


fashion_data = []
with open(os.path.join(path, 'fashion.json'), 'r') as f:
    for line in f:
        try:
            fashion_data.append(json.loads(line.strip()))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")


def convert_to_url(signature):
    return f'http://i.pinimg.com/400x/{signature[:2]}/{signature[2:4]}/{signature[4:6]}/{signature}.jpg'

# Image transformation for ResNet-50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ResNet-50 without the final classification layer
resnet50 = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
resnet50.eval()

# Extract 2048-dimensional feature vector from pool5
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        return resnet50(img_tensor).view(1, -1)


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.nn.functional.normalize(self.fc2(x), p=2, dim=1)


feedforward_net = FeedForwardNet(input_dim=2048, output_dim=128)

# Process images and obtain embeddings
def get_style_embedding(image_url, category):
    img_path = download_image(image_url, category)
    if img_path is None:
        return None
    features = extract_features(img_path)
    return feedforward_net(features)

embeddings = []
subset_size = 10000
with tqdm(total=subset_size, desc="Processing subset of fashion data") as pbar:
    for item in fashion_data[:subset_size]:
        product_url = convert_to_url(item['product'])
        scene_url = convert_to_url(item['scene'])
        product_embedding = get_style_embedding(product_url, category="product")
        scene_embedding = get_style_embedding(scene_url, category="scene")
        if product_embedding is not None and scene_embedding is not None:
            embeddings.append({
                "product_id": item["product"],
                "scene_id": item["scene"],
                "product_embedding": product_embedding.squeeze().tolist(),
                "scene_embedding": scene_embedding.squeeze().tolist()
            })
        pbar.update(1)

with open("embeddings.json", "w") as f:
    json.dump(embeddings, f)
torch.save(embeddings, "embeddings.pt")
