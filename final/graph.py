import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


"""
The KNN graph represents the relationships between images based on the similarity of their feature embeddings,
with nodes as images and edges connecting each image to its k nearest neighbors. 
If the feature embeddings effectively capture visual or stylistic properties, similar styles should cluster together, 
forming dense subgraphs in the graph. The visualization, using techniques like t-SNE, should show nodes with similar styles grouped closely,
and clusters can be validated through manual inspection or clustering analysis (e.g., K-Means). If the clustering aligns with visual similarities, 
it confirms the effectiveness of the feature extractor and graph construction.
"""

def load_features_and_filenames(feature_file, filename_file):
    """
    Load extracted features and filenames from files.
    :param feature_file: Path to the NumPy file containing features.
    :param filename_file: Path to the NumPy file containing filenames.
    :return: Tuple of features (NumPy array) and filenames (list).
    """
    features = np.load(feature_file)
    filenames = np.load(filename_file, allow_pickle=True)
    return features, filenames

def create_pyg_knn_graph(features, k=5):
    """
    Create a PyG graph from KNN relationships.
    :param features: NumPy array of feature embeddings.
    :param k: Number of nearest neighbors to connect.
    :return: PyTorch Geometric Data object.
    """
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(features, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data

def visualize_graph(graph_data, features, num_clusters=5, reduction_method='tsne'):
    """
    Visualize the PyG graph with nodes colored based on features or clustering.
    :param graph_data: PyTorch Geometric Data object.
    :param features: NumPy array of feature embeddings.
    :param num_clusters: Number of clusters for coloring (if clustering is used).
    :param reduction_method: Dimensionality reduction method ('tsne').
    """

    if reduction_method == 'tsne':
        reduced_features = TSNE(n_components=2, random_state=42).fit_transform(features)
    else:
        raise ValueError("Only 'tsne' is supported in this example.")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    pos = {i: reduced_features[i] for i in range(graph_data.num_nodes)}

    import networkx as nx
    edge_list = graph_data.edge_index.t().numpy()
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge_list)

    plt.figure(figsize=(12, 10))
    nx.draw(
        nx_graph, pos,
        node_size=50,
        node_color=cluster_labels,
        cmap=plt.cm.get_cmap('viridis', num_clusters),
        with_labels=False,
        alpha=0.7,
        edge_color="gray"
    )
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label="Cluster", ax=plt.gca())
    plt.title("KNN Graph Visualization with Feature-Based Coloring")
    plt.show()



def save_graph(graph_data, save_path="graph_full.pt"):
    """
    Save PyG Data object to a file.
    :param graph_data: PyTorch Geometric Data object.
    :param save_path: File path to save the graph data.
    """
    torch.save(graph_data, save_path)
    print(f"Graph saved to {save_path}")


if __name__ == '__main__':
    feature_file = "image_features_full.npy"
    filename_file = "image_filenames_full.npy"

    features, filenames = load_features_and_filenames(feature_file, filename_file)
    k = 5
    pyg_graph = create_pyg_knn_graph(features, k=k)
    visualize_graph(pyg_graph, features, num_clusters=5)

    save_graph(graph_data=pyg_graph)

