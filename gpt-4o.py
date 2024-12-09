import os
from openai import OpenAI

client = OpenAI(api_key="")
from PIL import Image
import json
from tqdm import tqdm

# Set your OpenAI API key

def generate_gpt4_caption(image_path):
    """
    Generate captions directly using GPT-4 with Vision or Text-based API.
    """
    try:
        # Open the image
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Call GPT-4 with Vision capabilities
        response = client.chat.completions.create(model="gpt-4o",  # Replace with the appropriate GPT-4 model name
        messages=[
            {"role": "system", "content": "You are an expert fashion analyst."},
            {"role": "user", "content": "Describe this clothing item in detail."}
        ],
        max_tokens=200)
        caption = response.choices[0].message.content
        return caption
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return None

if __name__ == '__main__':
    # Define image folder paths
    image_folders = [
        "images/product/",
        "images/scene/"
    ]

    captions = {}
    for folder in image_folders:
        if not os.path.exists(folder):
            print(f"Folder does not exist: {folder}")
            continue

        # Get all image files in the folder
        image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Generate captions for each image
        for image_path in tqdm(image_files, desc=f"Processing images in {folder}"):
            caption = generate_gpt4_caption(image_path)
            if caption:
                captions[image_path] = caption

    # Save captions to JSON
    with open("captions.json", "w") as f:
        json.dump(captions, f, indent=2)
    print("Captions saved to captions.json")
