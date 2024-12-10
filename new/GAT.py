import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# Step 1: Load PyG Data Object
def load_graph(load_path="graph_data.pt"):
    """
    Load PyG Data object from a file.
    :param load_path: File path to load the graph data from.
    :return: PyTorch Geometric Data object.
    """
    graph_data = torch.load(load_path)
    print(f"Graph loaded from {load_path}")
    return graph_data

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

# Step 3: Train GAT Model
def train_gat(graph_data, input_dim, hidden_dim, output_dim, epochs=100, lr=0.01):
    """
    Train a GAT model on the given graph data.
    :param graph_data: PyG Data object.
    :param input_dim: Dimension of input features.
    :param hidden_dim: Dimension of hidden layers.
    :param output_dim: Number of output classes.
    :param epochs: Number of training epochs.
    :param lr: Learning rate for the optimizer.
    """
    model = GAT(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Assign dummy labels for training (replace with real labels if available)
    labels = torch.randint(0, output_dim, (graph_data.num_nodes,))
    graph_data.y = labels

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = criterion(out, graph_data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    print("Training complete.")
    return model

# Step 4: Save Model
def save_gat_model(model, save_path="trained_gat_model.pt"):
    """
    Save the trained GAT model to a file.
    :param model: Trained GAT model.
    :param save_path: File path to save the model.
    """
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")

# Step 5: Main Function
if __name__ == '__main__':
    # Path to saved graph
    graph_path = "graph.pt"

    # Load graph data
    print("Loading graph...")
    graph_data = load_graph(graph_path)

    # Model dimensions
    input_dim = graph_data.x.size(1)  # Feature dimension
    hidden_dim = 16
    output_dim = 5  # Number of output classes/clusters

    # Train the GAT model
    print("Training GAT model...")
    trained_model = train_gat(graph_data, input_dim, hidden_dim, output_dim, epochs=500, lr=0.01)

    # Save the trained GAT model
    save_path = "trained_gat_model.pt"
    print("Saving trained GAT model...")
    save_gat_model(trained_model, save_path)
