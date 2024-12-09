import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import json
import numpy as np

from torch_geometric.data import Data

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=64, heads=1):
        super(GAT, self).__init__()
        from torch_geometric.nn import GATConv
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

def load_graph(graph_path="graph_data.pt"):
    data = torch.load(graph_path)
    return data

def load_model(model_path="model_weights.pth", input_dim=512, hidden_dim=64, output_dim=64, heads=1, device="cpu"):
    model = GAT(input_dim, hidden_dim, output_dim, heads)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_clip_model(model_name="openai/clip-vit-base-patch32", device="cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def encode_text(text, model, processor, device="cpu"):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.squeeze(0).cpu().numpy()

def get_node_embeddings(model, data, device="cpu"):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        z = model(x, edge_index)
    return z.cpu().numpy()

def load_original_mappings():
    with open("node_mappings.json", "r") as f:
        mappings = json.load(f)

    product_ids = mappings["product_ids"]  # product_ids in node order
    return product_ids

def recommend_products(user_query, model, data, clip_model, clip_processor, product_ids, top_k=5, device="cpu"):
    user_text_emb = encode_text(user_query, clip_model, clip_processor, device=device)
    node_embeddings = get_node_embeddings(model, data, device=device)

    # product nodes assumed from 0..len(product_ids)-1
    product_node_embeddings = node_embeddings[:len(product_ids)]

    user_text_emb_2d = user_text_emb.reshape(1, -1)
    sims = cosine_similarity(user_text_emb_2d, product_node_embeddings).flatten()

    top_indices = np.argsort(sims)[-top_k:][::-1]
    recommended_product_ids = [product_ids[i] for i in top_indices]

    return recommended_product_ids

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data, model, CLIP, and mappings
    data = load_graph("graph_data.pt")
    model = load_model("model_weights.pth", input_dim=data.x.size(1), hidden_dim=64, output_dim=64, heads=1, device=device)
    clip_model, clip_processor = load_clip_model(device=device)
    product_ids = load_original_mappings()

    # Read 20 prompts from a separate file
    prompts_file = "user_prompts.txt"
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    # Assume exactly 20 prompts (or handle if fewer/more)
    # prompts[i] is the user prompt for prompt_id i
    recommendations_dict = {}
    for i, prompt_text in enumerate(prompts):
        recommended_products = recommend_products(prompt_text, model, data, clip_model, clip_processor, product_ids, top_k=5, device=device)
        recommendations_dict[i] = recommended_products

    # Save recommendations to a JSON file
    with open("recommendations.json", "w") as f:
        json.dump(recommendations_dict, f, indent=2)

    print("Recommendations saved to recommendations.json")
