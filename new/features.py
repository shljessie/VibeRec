from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os 

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image to ensure consistent dimensions.
    :param image_path: Path to the image file.
    :param target_size: Desired size for the image (width, height).
    :return: Preprocessed PIL image.
    """
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        image = image.resize(target_size, Image.ANTIALIAS)  # Resize image
        return image
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image {image_path}: {e}")

def extract_features(image_path, processor, model, target_size=(224, 224)):
    """
    Extract features for a single image.
    :param image_path: Path to the image file.
    :param processor: Hugging Face image processor.
    :param model: Pretrained model for feature extraction.
    :param target_size: Target size for image resizing.
    :return: Extracted feature vector.
    """
    try:
        # Preprocess image
        image = preprocess_image(image_path, target_size)

        # Use processor to prepare input
        inputs = processor(images=image, return_tensors="pt")  # Wrap image in a list

        # Extract features using the model
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.logits[0].numpy()  # Access single batch element
        return features
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {image_path}: {e}")

def process_directory(directory_path, processor, model, limit=None):
    """
    Process all images in a directory and extract features.
    :param directory_path: Path to the directory containing images.
    :param processor: Hugging Face image processor.
    :param model: Pretrained model for feature extraction.
    :param limit: Optional limit for the number of images to process.
    :return: Tuple of feature list and filenames.
    """
    features_list = []
    filenames = []
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if limit is None:
        limit = len(image_files)

    for filename in tqdm(image_files[:limit], desc="Processing Images", unit="image"):
        image_path = os.path.join(directory_path, filename)
        try:
            features = extract_features(image_path, processor, model)
            features_list.append(features)
            filenames.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue  # Skip problematic files

    return np.array(features_list), filenames

if __name__ == '__main__':
    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
    model = AutoModelForImageClassification.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
    model.eval()

    # Directory containing images
    image_directory = "../images/product"

    # Process the directory and extract features
    features, filenames = process_directory(image_directory, processor, model)

    # Save features and filenames for future use
    np.save("image_features_full.npy", features)
    np.save("image_filenames_full.npy", filenames)

    print(f"Features extracted and saved. Total images processed: {len(filenames)}")
