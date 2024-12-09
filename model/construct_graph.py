# graph_construction.py
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

def create_graph(embeddings_path="clip_embeddings_with_text.pt"):
    # Load embeddings
    embeddings = torch.load(embeddings_path)

    # We'll create four categories of nodes:
    # - product nodes
    # - scene nodes
    # - product_text nodes
    # - scene_text nodes

    # We need to assign a unique node index for each product, scene, product_text, scene_text.
    product_to_idx = {}
    scene_to_idx = {}
    product_text_to_idx = {}
    scene_text_to_idx = {}

    product_count = 0
    scene_count = 0
    product_text_count = 0
    scene_text_count = 0

    # Collect features in arrays (we'll combine them later)
    product_features = []
    scene_features = []
    product_text_features = []
    scene_text_features = []

    # Edges: we will store them in lists, then convert to tensors
    product_scene_edges = []
    product_product_text_edges = []
    scene_scene_text_edges = []

    # Iterate over embeddings to populate the mappings
    for item in tqdm(embeddings, desc="Creating graph nodes and edges"):
        product_id = item["product_id"]
        scene_id = item["scene_id"]

        # Get embeddings
        p_emb = np.array(item["product_embedding"], dtype=np.float32)
        s_emb = np.array(item["scene_embedding"], dtype=np.float32)
        pt_emb = np.array(item["product_text_embedding"], dtype=np.float32) if item["product_text_embedding"] is not None else None
        st_emb = np.array(item["scene_text_embedding"], dtype=np.float32) if item["scene_text_embedding"] is not None else None

        # Assign product node index if not present
        if product_id not in product_to_idx:
            product_to_idx[product_id] = product_count
            product_features.append(p_emb)
            product_count += 1

        # Assign scene node index if not present
        if scene_id not in scene_to_idx:
            scene_to_idx[scene_id] = scene_count
            scene_features.append(s_emb)
            scene_count += 1

        p_idx = product_to_idx[product_id]
        s_idx = scene_to_idx[scene_id]

        # Create product_text node if present and not already created for that product_id
        # We assume one product_text node per product_id (if available)
        if pt_emb is not None and product_id not in product_text_to_idx:
            product_text_to_idx[product_id] = product_text_count
            product_text_features.append(pt_emb)
            product_text_count += 1

        # Create scene_text node if present and not already created for that scene_id
        # We assume one scene_text node per scene_id (if available)
        if st_emb is not None and scene_id not in scene_text_to_idx:
            scene_text_to_idx[scene_id] = scene_text_count
            scene_text_features.append(st_emb)
            scene_text_count += 1

        # Now we create edges:
        # product <-> scene
        product_scene_edges.append((p_idx, s_idx))
        product_scene_edges.append((s_idx, p_idx))  # undirected

        # product <-> product_text if product_text exists
        if product_id in product_text_to_idx:
            pt_idx = product_text_to_idx[product_id]
            # product_text nodes come after all product and scene nodes, so we must offset indexes later
            # We'll handle indexing after we know the counts.

        # scene <-> scene_text if scene_text exists
        if scene_id in scene_text_to_idx:
            st_idx = scene_text_to_idx[scene_id]
            # same note on indexing

    # Combine all node features into a single feature matrix
    # Node index arrangement:
    # products: [0 .. product_count-1]
    # scenes: [product_count .. product_count+scene_count-1]
    # product_text: [product_count+scene_count .. product_count+scene_count+product_text_count-1]
    # scene_text:   [product_count+scene_count+product_text_count .. end]

    scene_offset = product_count
    product_text_offset = product_count + scene_count
    scene_text_offset = product_count + scene_count + product_text_count

    # Convert lists to tensors
    product_features = torch.tensor(product_features, dtype=torch.float32) if product_features else torch.empty(0, 512)
    scene_features = torch.tensor(scene_features, dtype=torch.float32) if scene_features else torch.empty(0, 512)
    product_text_features = torch.tensor(product_text_features, dtype=torch.float32) if product_text_features else torch.empty(0, 512)
    scene_text_features = torch.tensor(scene_text_features, dtype=torch.float32) if scene_text_features else torch.empty(0, 512)

    x = torch.cat([product_features, scene_features, product_text_features, scene_text_features], dim=0)

    # Now add edges for product_text and scene_text
    # We must go through again since we need final indexes now
    product_product_text_edges = []
    scene_scene_text_edges = []
    for item in embeddings:
        product_id = item["product_id"]
        scene_id = item["scene_id"]

        if product_id in product_to_idx and product_id in product_text_to_idx:
            p_idx = product_to_idx[product_id]
            pt_idx = product_text_to_idx[product_id] + product_text_offset
            product_product_text_edges.append((p_idx, pt_idx))
            product_product_text_edges.append((pt_idx, p_idx))

        if scene_id in scene_to_idx and scene_id in scene_text_to_idx:
            s_idx = scene_to_idx[scene_id] + scene_offset
            st_idx = scene_text_to_idx[scene_id] + scene_text_offset
            scene_scene_text_edges.append((s_idx, st_idx))
            scene_scene_text_edges.append((st_idx, s_idx))

    # Adjust product_scene_edges indices for scenes
    # Scenes start at scene_offset
    new_product_scene_edges = []
    for (p_i, s_i) in product_scene_edges:
        # if p_i < product_count then p_i is product, s_i is scene index
        # scene idx should be s_i + scene_offset if s_i is a scene index
        # We must detect which is product and which is scene.
        # Actually, we know products < product_count, scenes < scene_count
        # Let's store them correctly:
        if p_i < product_count and s_i < scene_count:
            # p_i is product node index, s_i is scene node index
            new_product_scene_edges.append((p_i, s_i + scene_offset))
        else:
            # s_i < product_count means it's reversed, but we know we appended pairs p->s and s->p
            # Let's handle the general case:
            if p_i < product_count:
                # p_i product
                # s_i scene idx must offset
                new_product_scene_edges.append((p_i, s_i + scene_offset))
            else:
                # p_i scene index
                # s_i product index
                new_product_scene_edges.append((p_i + scene_offset, s_i))

    product_scene_edges = new_product_scene_edges

    # Combine all edges
    all_edges = product_scene_edges + product_product_text_edges + scene_scene_text_edges
    # all_edges is a list of (src, dst)
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    # Regarding connecting text nodes to each other: 
    # You mentioned it would create a cycle. Generally, connecting product_text <-> scene_text 
    # might overcomplicate the graph and isn't strictly necessary. 
    # The purpose of text nodes is to provide an additional modality anchor. 
    # It's probably best not to connect text-to-text directly at this stage.
    # If you want to experiment, you could add edges text-to-text, but let's skip that for now.

    return data

if __name__ == "__main__":
    data = create_graph("clip_embeddings_with_text.pt")
    torch.save(data, "graph_data.pt")
    print("Graph construction done. Data saved to graph_data.pt")
