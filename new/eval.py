import torch
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random
import os
from PIL import Image
import matplotlib.pyplot as plt


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        """
        Define a GAT model with two layers.
        :param input_dim: Dimension of input features.
        :param hidden_dim: Dimension of hidden layers.
        :param output_dim: Number of output classes.
        :param heads: Number of attention heads.
        """
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, data):
        """
        Forward pass of the GAT model.
        :param data: PyTorch Geometric Data object containing graph structure and features.
        :return: Log softmax of node classifications.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Step 1: Load Trained GAT Model
def load_gat_model(model_path="trained_gat_model.pt"):
    """
    Load the trained GAT model from a file.
    :param model_path: File path to the saved model.
    :return: Loaded GAT model.
    """
    model = torch.load(model_path)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# Step 2: Select Random Images
def select_random_images(feature_embeddings, num_samples=10):
    """
    Select random images (or products) from the feature embeddings.
    :param feature_embeddings: Numpy array or tensor of image embeddings.
    :param num_samples: Number of random samples to select.
    :return: Indices and selected embeddings.
    """
    random.seed(42)
    num_images = feature_embeddings.shape[0]
    random_indices = random.sample(range(num_images), num_samples)
    selected_embeddings = feature_embeddings[random_indices]
    return random_indices, selected_embeddings

# Step 3: Make Recommendations
def get_recommendations(model, graph_data, selected_indices):
    """
    Use the trained GAT model to get recommendations for selected images/products.
    :param model: Trained GAT model.
    :param graph_data: PyG Data object containing graph data.
    :param selected_indices: Indices of the selected images/products.
    :return: Predicted recommendations for each selected image/product.
    """
    with torch.no_grad():
        predictions = model(graph_data)
        recommended_indices = predictions[selected_indices].topk(5, dim=1).indices  # Top 5 recommendations
    return recommended_indices

def visualize_recommendations(selected_indices, recommendations, image_directory, filenames, save_dir="clustering_recs"):
    """
    Visualize the selected images and their recommendations.
    :param selected_indices: List of indices for the selected query images.
    :param recommendations: List of recommended indices for each query image.
    :param image_directory: Directory where images are stored.
    :param filenames: List of image filenames corresponding to embeddings.
    """

    # Create directory for saving visualizations
    os.makedirs(save_dir, exist_ok=True)

    for i, query_idx in enumerate(selected_indices):
        # Load query image
        query_image_path = os.path.join(image_directory, filenames[query_idx])
        query_image = Image.open(query_image_path)

        # Load recommended images
        recommended_image_paths = [os.path.join(image_directory, filenames[rec_idx]) for rec_idx in recommendations[i]]
        recommended_images = [Image.open(img_path) for img_path in recommended_image_paths]

        # Plot query and recommended images
        plt.figure(figsize=(15, 5))
        plt.subplot(1, len(recommended_images) + 1, 1)
        plt.imshow(query_image)
        plt.title("Query Image")
        plt.axis('off')

        for j, rec_image in enumerate(recommended_images, start=2):
            plt.subplot(1, len(recommended_images) + 1, j)
            plt.imshow(rec_image)
            plt.title(f"Recommendation {j - 1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # save the plot
        save_path = os.path.join(save_dir, f"similar_images_query_{query_idx}.png")
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close()


# Step 4: Main Function
if __name__ == '__main__':
    # Load trained GAT model
    trained_model_path = "clustering_gat_model.pt"
    trained_model = load_gat_model(trained_model_path)
    filenames = np.load("image_filenames_full.npy", allow_pickle=True) 
    image_directory = "../images/product" 

    # Load graph data
    graph_data_path = "graph_full.pt"
    graph_data = torch.load(graph_data_path)

    # Assuming `graph_data.x` contains image embeddings
    image_embeddings = graph_data.x.numpy()

    # Select 10 random images
    print("Selecting random images...")
    selected_indices, selected_embeddings = select_random_images(image_embeddings, num_samples=10)
    print(f"Selected indices: {selected_indices}")

    # Get recommendations
    print("Getting recommendations...")
    recommendations = get_recommendations(trained_model, graph_data, selected_indices)

    # Print recommendations
    for idx, recs in zip(selected_indices, recommendations):
        print(f"Image {idx} recommendations: {recs.tolist()}")

    # Visualize recommendations
    print("Visualizing recommendations...")
    visualize_recommendations(selected_indices, recommendations, image_directory, filenames, save_dir="clustering_recs_unsupervised")

