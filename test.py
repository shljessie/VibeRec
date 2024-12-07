# Save the trained model
torch.save(model.state_dict(), "gat_model.pth")

# Load the model for testing
model = GATModel(in_channels=graph.num_node_features, out_channels=128)
model.load_state_dict(torch.load("gat_model.pth"))
model.eval()
