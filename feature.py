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
import random

print(torch.backends.mps.is_available())

def download_image(url, category, base_dir="images"):
    # Define the save path based on the category ("scene" or "product")
    save_dir = os.path.join(base_dir, category)
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract the image name from the URL and define the local save path
    image_name = url.split("/")[-1]
    save_path = os.path.join(save_dir, image_name)
    
    # Download the image
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return save_path
    else:
        print(f"Failed to download image: {url}")
        return None

# Step 1: Download the dataset
path = kagglehub.dataset_download("pypiahmad/shop-the-look-dataset")
print("Path to dataset files:", path)

# Step 2: Parse JSON and load images
fashion_data = []

# Read each line as a JSON object
with open(os.path.join(path, 'fashion.json'), 'r') as f:
    for line in f:
        try:
            # Parse each line as a JSON object
            fashion_data.append(json.loads(line.strip()))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

subset_size = 10000  

# Function to create image URL
def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)

# Define transformation for ResNet-50 input
# transforms.Compose to chain multiple image transformations together
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalize based on imagenet's mean and std
])

# Step 3: Load ResNet-50 and extract features

# Load ResNet-50 model and modify it to remove the final classification layer
resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])  # Removes the last FC layer, which is the classification layer, * passes each layer as a separate argumemnt
resnet50.eval() #inference mode

# Updated extract_features function
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) # add an additional dimension, at the specificed position  - adding the batch dimension
    
    # Extract 2048-dimensional feature vector from pool5
    with torch.no_grad():
        pool5_features = resnet50(img_tensor)
        pool5_features = pool5_features.view(pool5_features.size(0), -1)  # Flatten to (1, 2048)
    
    return pool5_features

# Now, `pool5_features` should have the correct shape of (1, 2048) to pass into the feedforward network

# Step 4: Define the two-layer feedforward network
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.normalize(self.fc2(x), p=2, dim=1)  # Normalize to unit length
        return x

# Initialize the feedforward network
feedforward_net = FeedForwardNet(input_dim=2048, output_dim=128)  # d-dimensional embedding

# Step 5: Process images and obtain metric embeddings
def get_style_embedding(image_url, category):
    # Download image to the specified category folder
    img_path = download_image(image_url, category=category)
    
    # Check if image was downloaded successfully
    if img_path is None:
        return None
    
    # Extract features from the image
    pool5_features = extract_features(img_path)
    
    # Transform to style space
    embedding = feedforward_net(pool5_features)
    return embedding

# Extract Embeddings with Overall Progress Bar for the subset
embeddings = []

with tqdm(total=subset_size, desc="Processing subset of fashion data") as pbar:
    for item in fashion_data[:subset_size]:
        product_url = convert_to_url(item['product'])
        scene_url = convert_to_url(item['scene'])

        # Get embeddings
        product_embedding = get_style_embedding(product_url, category="product")
        scene_embedding = get_style_embedding(scene_url, category="scene")

        if product_embedding is not None and scene_embedding is not None:
            embeddings.append({
                "product_id": item["product"],  # Store the unique product ID
                "scene_id": item["scene"],      # Store the unique scene ID
                "product_embedding": product_embedding.squeeze().tolist(),
                "scene_embedding": scene_embedding.squeeze().tolist()
            })
        
        pbar.update(1)


import json

with open("embeddings.json", "w") as f:
    json.dump(embeddings, f)


torch.save(embeddings, "embeddings.pt")
