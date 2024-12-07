import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.utils.parallel import joblib
joblib.parallel_backend("threading")

# Load embeddings from the .pt file
embeddings_data = torch.load("embeddings.pt")

# Extract embeddings and create labels
embeddings = []
labels = []
for item in embeddings_data:
    # Add product embedding with "product" label
    embeddings.append(item["product_embedding"])
    labels.append("product")

    # Add scene embedding with "scene" label
    embeddings.append(item["scene_embedding"])
    labels.append("scene")

# Convert embeddings to torch.Tensor
embeddings = torch.tensor(embeddings, dtype=torch.float)

# Visualization function
def visualize_embeddings(embeddings, labels, method="tsne"):
    """
    Visualize node embeddings in 2D space.
    Args:
        embeddings (torch.Tensor): Node embeddings.
        labels (list): List of labels for each node (e.g., "scene" or "product").
        method (str): Dimensionality reduction method ("tsne" or "pca").
    """
    embeddings = embeddings.detach().numpy()
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported method. Use 'tsne' or 'pca'.")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    for label in set(labels):
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            reduced_embeddings[idx, 0],
            reduced_embeddings[idx, 1],
            label=label,
            alpha=0.7,
        )
    plt.title(f"Node Embeddings Visualization ({method.upper()})")
    plt.legend()
    plt.show()

# Example visualization
visualize_embeddings(embeddings, labels, method="tsne")
