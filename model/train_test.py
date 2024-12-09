# train_test.py
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.nn import GATConv

########################
# Model Definition
########################
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=64, heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

########################
# Data and Split
########################
def train_test_split_edges(data, test_ratio=0.1):
    # We'll only create train and test sets
    edges = data.edge_index.t().tolist()
    total_edges = len(edges)
    test_count = int(total_edges * test_ratio)

    # Shuffle and split
    torch.manual_seed(42)
    edges = torch.randperm(total_edges)
    test_edges_idx = edges[:test_count]
    train_edges_idx = edges[test_count:]

    all_edges = data.edge_index.t()
    train_edge_index = all_edges[train_edges_idx].t().contiguous()
    test_edge_index = all_edges[test_edges_idx].t().contiguous()

    return train_edge_index, test_edge_index

def get_link_labels(pos_edge, neg_edge):
    E_pos = pos_edge.size(1)
    E_neg = neg_edge.size(1)
    labels = torch.cat([torch.ones(E_pos), torch.zeros(E_neg)], dim=0)
    return labels

########################
# Training Loop
########################
def train(model, x, train_pos_edge_index, optimizer):
    model.train()
    optimizer.zero_grad()

    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index,
        num_nodes=x.size(0),
        num_neg_samples=train_pos_edge_index.size(1)
    )

    z = model(x, train_pos_edge_index)

    pos_scores = (z[train_pos_edge_index[0]] * z[train_pos_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = get_link_labels(train_pos_edge_index, neg_edge_index).to(scores.device)

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

########################
# Evaluation
########################
@torch.no_grad()
def test(model, x, test_pos_edge_index, train_pos_edge_index):
    model.eval()
    # For inference, we use the full set of edges (train+test) for embedding computation
    # It's common in link prediction to use the full graph for embedding generation
    # Here, we just use train_pos_edge_index (or you could use the full edge_index if you wish)
    z = model(x, train_pos_edge_index)

    # Negative sampling for test
    neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index,  # use train edges for neg sampling or the full graph if preferred
        num_nodes=x.size(0),
        num_neg_samples=test_pos_edge_index.size(1)
    )

    pos_scores = (z[test_pos_edge_index[0]] * z[test_pos_edge_index[1]]).sum(dim=1).cpu()
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()

    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = get_link_labels(test_pos_edge_index, neg_edge_index).numpy()

    # Compute AUC and AP
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

########################
# Main
########################
if __name__ == "__main__":
    data = torch.load("graph_data.pt")

    # Split into train/test edges
    train_edge_index, test_edge_index = train_test_split_edges(data, test_ratio=0.1)

    model = GAT(input_dim=data.x.size(1), hidden_dim=64, output_dim=64, heads=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    train_edge_index = train_edge_index.to(device)
    test_edge_index = test_edge_index.to(device)
    model = model.to(device)

    # We'll just train for a fixed number of epochs and then evaluate
    for epoch in range(1, 101):
        loss = train(model, data.x, train_edge_index, optimizer)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate
    auc, ap = test(model, data.x, test_edge_index, train_edge_index)
    print(f"Test AUC: {auc:.4f}, Test AP: {ap:.4f}")

    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved to model_weights.pth")
