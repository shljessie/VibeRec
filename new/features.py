import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

def extract_features(image_path, processor, model):
    """
    Extract features for a single image.
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(**inputs)
        # Access the feature vector (e.g., logits or pooled output)
        features = outputs.logits.squeeze().numpy()  # Adjust based on your needs
    return features

def process_directory(directory_path, processor, model, limit=1000):
    """
    Process all images in a directory and extract features.
    """
    features_list = []
    filenames = []
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for filename in tqdm(image_files[:limit], desc="Processing Images", unit="image"):
        image_path = os.path.join(directory_path, filename)
        try:
            features = extract_features(image_path, processor, model)
            features_list.append(features)
            filenames.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return np.array(features_list), filenames

if __name__ == '__main__':
    # Load the image processor and model
    processor = AutoImageProcessor.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
    model = AutoModelForImageClassification.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")   #https://huggingface.co/arize-ai/resnet-50-fashion-mnist-quality-drift

    # Remove the classification head for feature extraction
    model.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Directory containing product and scene images
    image_directory = "../images/product"
    features, filenames = process_directory(image_directory, processor, model)

    # Save features to a file for later use
    np.save("image_features.npy", features)
    np.save("image_filenames.npy", filenames)

    print(f"Features extracted and saved. Total images processed: {len(filenames)}")
