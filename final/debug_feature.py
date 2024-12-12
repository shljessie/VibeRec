import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
model = AutoModelForImageClassification.from_pretrained("arize-ai/resnet-50-fashion-mnist-quality-drift")
model.eval()

# Create a dummy RGB image tensor (3 channels, 224x224)
dummy_image = torch.rand(1, 3, 224, 224)  # Batch size 1, 3 channels (RGB), 224x224 resolution

# Debug the processor with the dummy image
try:
    inputs = processor(images=dummy_image, return_tensors="pt")
    print("Processor input keys:", inputs.keys())
    print("Input tensor shape:", inputs['pixel_values'].shape)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
        print("Model output shape:", outputs.logits.shape)
except Exception as e:
    print("Error while debugging with dummy image:", e)
