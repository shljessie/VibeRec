import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Step 2: Define GAT Model
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
    
# Metric Functions
def compute_silhouette_score(features, cluster_labels):
    """
    Compute Silhouette Score for clustering.
    :param features: Node feature embeddings (NumPy array).
    :param cluster_labels: Cluster labels for each node.
    :return: Silhouette score.
    """
    return silhouette_score(features, cluster_labels)

def compute_davies_bouldin_index(features, cluster_labels):
    """
    Compute Davies-Bouldin Index for clustering.
    :param features: Node feature embeddings (NumPy array).
    :param cluster_labels: Cluster labels for each node.
    :return: Davies-Bouldin index.
    """
    return davies_bouldin_score(features, cluster_labels)

def compute_graph_density(graph_data):
    """
    Compute density of the graph.
    :param graph_data: PyG Data object.
    :return: Graph density.
    """
    num_edges = graph_data.edge_index.size(1)
    num_nodes = graph_data.num_nodes
    max_edges = num_nodes * (num_nodes - 1)  # For an undirected graph
    return num_edges / max_edges

def plot_degree_distribution(graph_data):
    """
    Plot the degree distribution of the graph.
    :param graph_data: PyG Data object.
    """
    degrees = torch.bincount(graph_data.edge_index[0])  # Node degrees
    plt.figure(figsize=(8, 6))
    plt.hist(degrees.numpy(), bins=20, edgecolor='black', alpha=0.7)
    plt.title("Node Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def evaluate_trained_gat_model(graph_data, model, num_clusters=5):
    """
    Evaluate the graph using embeddings from a trained GAT model.
    :param graph_data: PyG Data object.
    :param model: Trained GAT model.
    :param num_clusters: Number of clusters for clustering-based metrics.
    """
    model.eval()
    with torch.no_grad():
        # Generate node embeddings using the trained GAT model
        embeddings = model(graph_data).detach().cpu().numpy()

    # Perform clustering
    print("Performing clustering with trained GAT model embeddings...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute metrics
    print("Computing metrics for trained GAT model...")
    silhouette = compute_silhouette_score(embeddings, cluster_labels)
    davies_bouldin = compute_davies_bouldin_index(embeddings, cluster_labels)

    print(f"Trained GAT Model Silhouette Score: {silhouette:.4f}")
    print(f"Trained GAT Model Davies-Bouldin Index: {davies_bouldin:.4f}")

    # Save metrics to a file
    with open("gat_clustering_model_metrics.txt", "w") as f:
        f.write(f"Trained GAT Model Silhouette Score: {silhouette:.4f}\n")
        f.write(f"Trained GAT Model Davies-Bouldin Index: {davies_bouldin:.4f}\n")

# Evaluation Script
def evaluate_graph(feature_file, filename_file, graph_file, num_clusters=5, trained_gat_model=None):
    """
    Evaluate the graph using various metrics.
    :param feature_file: Path to the NumPy file containing features.
    :param filename_file: Path to the NumPy file containing filenames.
    :param graph_file: Path to the saved PyG graph file.
    :param num_clusters: Number of clusters for clustering-based metrics.
    :param trained_gat_model: Optional trained GAT model for additional evaluation.
    """
    # Load features, filenames, and graph data
    features = np.load(feature_file)
    filenames = np.load(filename_file, allow_pickle=True)
    graph_data = torch.load(graph_file)

    # Perform clustering
    print("Performing clustering with raw features...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Compute metrics
    print("Computing metrics for raw features...")
    silhouette = compute_silhouette_score(features, cluster_labels)
    davies_bouldin = compute_davies_bouldin_index(features, cluster_labels)
    density = compute_graph_density(graph_data)

    print(f"Raw Features Silhouette Score: {silhouette:.4f}")
    print(f"Raw Features Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"Graph Density: {density:.4f}")

    # Plot degree distribution
    print("Plotting degree distribution...")
    plot_degree_distribution(graph_data)

    # Save metrics to a file
    with open("graph_metrics.txt", "w") as f:
        f.write(f"Raw Features Silhouette Score: {silhouette:.4f}\n")
        f.write(f"Raw Features Davies-Bouldin Index: {davies_bouldin:.4f}\n")
        f.write(f"Graph Density: {density:.4f}\n")

    # Evaluate with trained GAT model if provided
    if trained_gat_model is not None:
        evaluate_trained_gat_model(graph_data, trained_gat_model, num_clusters)

if __name__ == '__main__':
    # Paths to input files
    feature_file = "image_features_full.npy"
    filename_file = "image_filenames_full.npy"
    graph_file = "graph_full.pt"

    # Number of clusters
    num_clusters = 5

    # Load a trained GAT model if available
    trained_gat_model = None
    try:
        trained_gat_model = torch.load("clustering_gat_model.pt")
    except FileNotFoundError:
        print("Trained GAT model not found. Skipping GAT evaluation.")

    # Run the evaluation
    evaluate_graph(feature_file, filename_file, graph_file, num_clusters, trained_gat_model)
