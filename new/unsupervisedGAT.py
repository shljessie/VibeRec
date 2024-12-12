import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def load_graph(load_path="graph_data.pt"):
    """
    Load PyG Data object from a file.
    :param load_path: File path to load the graph data from.
    :return: PyTorch Geometric Data object.
    """
    graph_data = torch.load(load_path)
    print(f"Graph loaded from {load_path}")
    return graph_data
# Step 1: Define Robust GAT Model for Clustering
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, heads=4, dropout=0.3):
        """
        Define a robust GAT model for clustering tasks.
        :param input_dim: Dimension of input features.
        :param hidden_dim: Dimension of hidden layers.
        :param embedding_dim: Dimension of output embeddings.
        :param heads: Number of attention heads.
        :param dropout: Dropout probability.
        """
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, embedding_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * heads)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, data):
        """
        Forward pass of the GAT model.
        :param data: PyTorch Geometric Data object containing graph structure and features.
        :return: Node embeddings.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.layer_norm1(self.gat1(x, edge_index)))
        x = self.dropout(x)
        x = self.layer_norm2(self.gat2(x, edge_index))
        return x  # Return embeddings for clustering

# Step 2: Train GAT Model for Clustering
def train_gat_clustering(graph_data, input_dim, hidden_dim, embedding_dim, num_clusters, epochs=100, lr=0.01):
    """
    Train a GAT model to produce cluster-friendly embeddings.
    :param graph_data: PyG Data object.
    :param input_dim: Dimension of input features.
    :param hidden_dim: Dimension of hidden layers.
    :param embedding_dim: Dimension of output embeddings.
    :param num_clusters: Number of clusters.
    :param epochs: Number of training epochs.
    :param lr: Learning rate for the optimizer.
    """
    model = GAT(input_dim, hidden_dim, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(graph_data)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
        cluster_assignments = kmeans.fit_predict(embeddings.detach().cpu().numpy())

        # Compute clustering-friendly loss
        loss = clustering_loss(embeddings, graph_data.edge_index, cluster_assignments)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    print("Training complete.")
    return model, kmeans

# Step 3: Clustering-Friendly Loss
def clustering_loss(embeddings, edge_index, cluster_assignments, alpha=0.8, beta=0.05):
    cluster_assignments = torch.tensor(cluster_assignments, device=embeddings.device)
    row, col = edge_index
    pos_sim = torch.sum(embeddings[row] * embeddings[col], dim=-1)
    pos_loss = -torch.log(torch.sigmoid(pos_sim) + 1e-15).mean()

    cluster_centers = torch.stack(
        [embeddings[cluster_assignments == c].mean(dim=0) for c in range(cluster_assignments.max().item() + 1)]
    )
    cluster_loss = torch.norm(embeddings - cluster_centers[cluster_assignments], dim=1).mean()

    # Embedding diversity regularization
    embedding_reg = torch.norm(embeddings, dim=1).mean()

    return alpha * pos_loss + (1 - alpha) * cluster_loss + beta * embedding_reg


# Step 4: Save Model
def save_gat_model(model, save_path="clustering_gat_model.pt"):
    """
    Save the trained GAT model to a file.
    :param model: Trained GAT model.
    :param save_path: File path to save the model.
    """
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    # Path to saved graph
    graph_path = "graph_full.pt"
    
    # Load graph data
    print("Loading graph...")
    graph_data = load_graph(graph_path)
    graph_data.x = torch.tensor(StandardScaler().fit_transform(graph_data.x.numpy()), dtype=torch.float32)
    
    # Model dimensions
    input_dim = graph_data.x.size(1)  # Feature dimension
    hidden_dim = 128  # Increase from 32
    embedding_dim = 256  # Increase from 64
    heads = 8 
    num_clusters = 5  # Number of clusters
    
    # Train the GAT model for clustering
    print("Training clustering GAT model...")
    trained_model, kmeans = train_gat_clustering(graph_data, input_dim, hidden_dim, embedding_dim, num_clusters, epochs=50, lr=0.001)
    
    # Save the trained GAT model
    save_path = "clustering_gat_model.pt"
    print("Saving trained GAT model...")
    save_gat_model(trained_model, save_path)

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    embeddings = trained_model(graph_data).detach().cpu().numpy()
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue')
    plt.title("Embedding Visualization")
    plt.show()