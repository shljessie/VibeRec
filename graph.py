import torch
from torch_geometric.data import Data
import numpy as np

def create_bipartite_graph_with_vibes(embeddings_path, vibe_embeddings):
    embeddings = torch.load(embeddings_path)

    node_features = {}
    edges = []

    # Add product and scene nodes
    for item in embeddings:
        product_id = item["product_id"]
        scene_id = item["scene_id"]

        if product_id not in node_features:
            node_features[product_id] = np.array(item["product_embedding"])
        if scene_id not in node_features:
            node_features[scene_id] = np.array(item["scene_embedding"])

        edges.append((product_id, scene_id))

    # Add vibe description nodes and edges
    for node_id in list(node_features.keys()):
        vibe_id = f"vibe_{node_id}"
        node_features[vibe_id] = vibe_embeddings[node_id]
        edges.append((node_id, vibe_id))  # Connect product/scene to its vibe
        edges.append((vibe_id, node_id))  # Ensure bidirectional connection

    # Convert to PyTorch Geometric format
    node_ids = list(node_features.keys())
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    edge_index = torch.tensor(
        [[node_id_to_index[src], node_id_to_index[dst]] for src, dst in edges],
        dtype=torch.long,
    ).t()

    feature_matrix = torch.tensor(
        [node_features[node_id] for node_id in node_ids], dtype=torch.float
    )

    return Data(x=feature_matrix, edge_index=edge_index, node_ids=node_ids)



# Implementation
# Create the graph
# bipartite_graph_with_vibes = create_bipartite_graph_with_vibes("embeddings.pt", vibe_embeddings)

# # Print the graph details
# print(bipartite_graph_with_vibes)