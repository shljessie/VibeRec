# graph_construction.py
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import json

def create_graph(embeddings_path="clip_embeddings_with_text.pt"):
    # Load embeddings
    embeddings = torch.load(embeddings_path)

    product_to_idx = {}
    scene_to_idx = {}
    product_text_to_idx = {}
    scene_text_to_idx = {}

    product_count = 0
    scene_count = 0
    product_text_count = 0
    scene_text_count = 0

    product_features = []
    scene_features = []
    product_text_features = []
    scene_text_features = []

    product_scene_edges = []

    for item in tqdm(embeddings, desc="Creating graph nodes and edges"):
        product_id = item["product_id"]
        scene_id = item["scene_id"]

        p_emb = np.array(item["product_embedding"], dtype=np.float32)
        s_emb = np.array(item["scene_embedding"], dtype=np.float32)
        pt_emb = np.array(item["product_text_embedding"], dtype=np.float32) if item["product_text_embedding"] is not None else None
        st_emb = np.array(item["scene_text_embedding"], dtype=np.float32) if item["scene_text_embedding"] is not None else None

        if product_id not in product_to_idx:
            product_to_idx[product_id] = product_count
            product_features.append(p_emb)
            product_count += 1

        if scene_id not in scene_to_idx:
            scene_to_idx[scene_id] = scene_count
            scene_features.append(s_emb)
            scene_count += 1

        p_idx = product_to_idx[product_id]
        s_idx = scene_to_idx[scene_id]

        # We'll store edges for product-scene now and adjust indexing later
        product_scene_edges.append((p_idx, s_idx))
        product_scene_edges.append((s_idx, p_idx))

        # Create product_text node if needed
        if pt_emb is not None and product_id not in product_text_to_idx:
            product_text_to_idx[product_id] = product_text_count
            product_text_features.append(pt_emb)
            product_text_count += 1

        # Create scene_text node if needed
        if st_emb is not None and scene_id not in scene_text_to_idx:
            scene_text_to_idx[scene_id] = scene_text_count
            scene_text_features.append(st_emb)
            scene_text_count += 1

    scene_offset = product_count
    product_text_offset = product_count + scene_count
    scene_text_offset = product_count + scene_count + product_text_count

    product_features = torch.tensor(product_features, dtype=torch.float32) if product_features else torch.empty(0, 512)
    scene_features = torch.tensor(scene_features, dtype=torch.float32) if scene_features else torch.empty(0, 512)
    product_text_features = torch.tensor(product_text_features, dtype=torch.float32) if product_text_features else torch.empty(0, 512)
    scene_text_features = torch.tensor(scene_text_features, dtype=torch.float32) if scene_text_features else torch.empty(0, 512)

    x = torch.cat([product_features, scene_features, product_text_features, scene_text_features], dim=0)

    # Recompute product_product_text_edges and scene_scene_text_edges now that indexing is known
    product_product_text_edges = []
    scene_scene_text_edges = []
    for item in embeddings:
        product_id = item["product_id"]
        scene_id = item["scene_id"]

        # product <-> product_text
        if product_id in product_to_idx and product_id in product_text_to_idx:
            p_idx = product_to_idx[product_id]
            pt_idx = product_text_to_idx[product_id] + product_text_offset
            product_product_text_edges.append((p_idx, pt_idx))
            product_product_text_edges.append((pt_idx, p_idx))

        # scene <-> scene_text
        if scene_id in scene_to_idx and scene_id in scene_text_to_idx:
            s_idx = scene_to_idx[scene_id] + scene_offset
            st_idx = scene_text_to_idx[scene_id] + scene_text_offset
            scene_scene_text_edges.append((s_idx, st_idx))
            scene_scene_text_edges.append((st_idx, s_idx))

    # Adjust product_scene_edges
    # Currently, product_scene_edges has (p_idx, s_idx) in product/scene index space
    # We must offset scene indices
    new_product_scene_edges = []
    for (a, b) in product_scene_edges:
        # a might be product idx (< product_count) or scene idx (< scene_count)
        # b likewise. We need to figure out who is product and who is scene.
        if a < product_count and b < scene_count:
            # a is product, b is scene
            new_product_scene_edges.append((a, b + scene_offset))
        elif b < product_count and a < scene_count:
            # b is product, a is scene
            new_product_scene_edges.append((b, a + scene_offset))
        else:
            # If we ended up here, double check logic. It's possible we are processing reversed edges.
            # The above conditions should cover both forward and backward edges.
            if a < product_count:
                new_product_scene_edges.append((a, b + scene_offset))
            else:
                new_product_scene_edges.append((b, a + scene_offset))

    product_scene_edges = new_product_scene_edges

    all_edges = product_scene_edges + product_product_text_edges + scene_scene_text_edges
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    # Create node mappings:
    # products: [0 .. product_count-1]
    # scenes: [product_count .. product_count+scene_count-1]
    # product_text: [product_count+scene_count .. product_count+scene_count+product_text_count-1]
    # scene_text: [product_count+scene_count+product_text_count .. end]

    # Sort product_ids by their assigned index (key: product_to_idx)
    sorted_products = sorted(product_to_idx.items(), key=lambda x: x[1])  # (product_id, idx)
    product_ids_ordered = [p[0] for p in sorted_products]

    sorted_scenes = sorted(scene_to_idx.items(), key=lambda x: x[1])
    scene_ids_ordered = [s[0] for s in sorted_scenes]

    # We could also store mappings for text nodes if needed
    # For now, just store product_ids and scene_ids. If you want text node mappings, do the same sorting.
    node_mappings = {
        "product_ids": product_ids_ordered,
        "scene_ids": scene_ids_ordered
        # If needed:
        # "product_text_ids": list(product_text_to_idx.keys()) sorted by idx
        # "scene_text_ids": list(scene_text_to_idx.keys()) sorted by idx
    }

    with open("node_mappings.json", "w") as f:
        json.dump(node_mappings, f, indent=2)

    return data

if __name__ == "__main__":
    data = create_graph("clip_embeddings_with_text.pt")
    torch.save(data, "graph_data.pt")
    print("Graph construction done. Data saved to graph_data.pt and node_mappings.json created.")
