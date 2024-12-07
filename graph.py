# graph.py
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from tqdm import tqdm

def create_graph():
    # Load embeddings
    embeddings = torch.load("embeddings.pt")

    # Create a dictionary to map IDs to embeddings
    node_features = {}
    for item in embeddings:
        node_features[item["product_id"]] = np.array(item["product_embedding"])
        node_features[item["scene_id"]] = np.array(item["scene_embedding"])

    # Prepare node features and IDs
    node_ids = list(node_features.keys())
    feature_matrix = np.array([node_features[node_id] for node_id in node_ids])

    # Use k-NN for edge creation
    k = 10  # Number of neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(feature_matrix)
    distances, indices = nbrs.kneighbors(feature_matrix)

    # Build edge index
    edges = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:  # Avoid self-loops
                edges.append((i, neighbor))

    # Convert edge list to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Convert node features to tensor
    node_features_tensor = torch.tensor(feature_matrix, dtype=torch.float)

    # Create the graph data object
    graph = Data(x=node_features_tensor, edge_index=edge_index)
    return graph
