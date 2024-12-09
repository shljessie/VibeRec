# Personalized Complete the Look

## Main Files
1. **Feature Extraction (`feature.py`):**
   - Extracts 2048-dimensional feature vectors from scene and product images using a pre-trained ResNet-50 model.
   - Reduces these feature vectors to 128-dimensional embeddings using a custom two-layer feedforward neural network (FFNN).

2. **Graph Representation (`graph.py`):**
   - Constructs a bipartite graph where nodes represent products or scenes with feature embeddings, and edges connect similar nodes using k-Nearest Neighbors (k-NN) based on cosine similarity.
   - Additionally, **"vibe description"** nodes are added for each product and scene, initialized with embeddings from a visual-language model (e.g., BLIP-2), and connected to their respective product or scene nodes.
   - The graph includes a feature matrix (x) for node embeddings and an edge index (edge_index) for connectivity, enabling tasks such as classification and recommendations using Graph Neural Networks (GNNs)

3. **Vibe Embeddings (`blip.py`):**
- This script generates "vibe description" captions for a subset of the fashion dataset using a BLIP-2 model. These captions describe the aesthetic and style of each product or scene image, focusing on fashion-specific elements like colors, textures, and mood. Here's the process broken down:
## Directory Structure
- `images/`: Contains the scene and product images used for feature extraction.
- `feature.py`: Code for feature extraction and embedding generation.
- `graph.py`: Code for graph construction using PyTorch Geometric.
- `gat.py`: Implements and trains the Graph Attention Network (GAT).

## Saved Files
- `gat_model.pth`: Trained GAT model weights.
- `embeddings.pt`: Generated 128-dimensional embeddings for all products and scenes.

3. **Access Results:**
   - The refined embeddings are saved to `embeddings.pt`.
   - The trained GAT model is saved to `gat_model.pth`.