# Personalized Complete the Look


## Main Files
1. **Feature Extraction (`feature.py`):**
   - Extracts 2048-dimensional feature vectors from scene and product images using a pre-trained ResNet-50 model.
   - Reduces these feature vectors to 128-dimensional embeddings using a custom two-layer feedforward neural network (FFNN).

2. **Graph Representation (`graph.py`):**
   - Constructs a graph where nodes represent products or scenes, and edges connect nodes based on similarity computed using k-Nearest Neighbors (k-NN).
   - Stores node features in a feature matrix (`x`) and relationships in an edge index tensor (`edge_index`).

3. **Graph Attention Network (`gat.py`):**
   - Refines embeddings using the Graph Attention Network (GAT) to capture contextual relationships.
   - Outputs enriched embeddings for downstream tasks like clustering or recommendations.

## Directory Structure
- `images/`: Contains the scene and product images used for feature extraction.
- `feature.py`: Code for feature extraction and embedding generation.
- `graph.py`: Code for graph construction using PyTorch Geometric.
- `gat.py`: Implements and trains the Graph Attention Network (GAT).

## Saved Files
- `gat_model.pth`: Trained GAT model weights.
- `embeddings.pt`: Generated 128-dimensional embeddings for all products and scenes.

## Usage
1. **Extract Features:**
   - Run `feature.py` to extract features and generate embeddings:
     ```bash
     python feature.py
     ```

2. **Create Graph:**
   - Use `graph.py` to construct the graph:
     ```bash
     python graph.py
     ```

3. **Train the GAT Model:**
   - Train the Graph Attention Network with `gat.py`:
     ```bash
     python gat.py
     ```

4. **Access Results:**
   - The refined embeddings are saved to `embeddings.pt`.
   - The trained GAT model is saved to `gat_model.pth`.

