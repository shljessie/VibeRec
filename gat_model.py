import torch
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from graph import create_graph

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)

        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))  
        x = self.gat2(x, edge_index)
        return x


graph = create_graph()

data_loader = DataLoader([graph], batch_size=1, shuffle=False)

# Initialize the GAT model
input_dim = graph.x.shape[1]  # 128-dimensional embeddings
hidden_dim = 64
output_dim = 32
gat_model = GAT(input_dim, hidden_dim, output_dim)

# Training setup
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Dummy target labels for illustration (e.g., clustering or classification)
# Replace with real labels in actual use
target = torch.randint(0, 3, (graph.x.size(0),))

# Training loop
def train(model, data_loader, target, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, target) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


train(gat_model, data_loader, target)

# Save the model after training
torch.save(gat_model.state_dict(), "gat_model.pth")

