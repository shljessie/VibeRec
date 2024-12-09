import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def create_graph():
    # Load embeddings
    embeddings = torch.load("embeddings.pt")

    # Map IDs to embeddings
    node_features = {
        item["product_id"]: np.array(item["product_embedding"])
        for item in embeddings
    }
    node_features.update({
        item["scene_id"]: np.array(item["scene_embedding"])
        for item in embeddings
    })

    node_ids = list(node_features.keys())
    feature_matrix = np.array([node_features[node_id] for node_id in node_ids])

    k = 10
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(feature_matrix)
    indices = nbrs.kneighbors(feature_matrix, return_distance=False)

    edges = [(i, neighbor) for i, neighbors in enumerate(indices) for neighbor in neighbors if i != neighbor]

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    node_features_tensor = torch.tensor(feature_matrix, dtype=torch.float)

    return Data(x=node_features_tensor, edge_index=edge_index)
