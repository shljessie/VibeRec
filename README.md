# Personalized Complete the Look

This project explores a novel approach to personalized fashion recommendation systems by leveraging user-curated visual collections (e.g., moodboards) and graph-based learning techniques. The system combines advanced feature extraction, graph construction, and Graph Attention Networks (GAT) to provide contextually relevant and aesthetically aligned recommendations.

To learn more about the project check out the Medium Article : https://medium.com/@jessielee.shl/viberec-fashion-recommendation-with-graph-neural-networks-d381703ef8a8

## Authors

- **Seonghee Lee** (shl1027)
- **Jeffrey Heo** (jeffheo)
- **Juliana Ma** (juliamma)

This project was developed for **CS 224W: Machine Learning with Graphs**.

---

## Key Components

### **1. Feature Extraction (`feature.py`)**

- **Process**: Product and scene images are passed through a fine-tuned ResNet-50 model, pre-trained on the Fashion MNIST dataset.
- **Output**: 2048-dimensional feature vectors extracted from the model’s penultimate layer, further reduced to 128-dimensional embeddings using a two-layer feedforward neural network (FFNN).
- **Implementation**: Utilizes Hugging Face’s ResNet-50 model (`arize-ai/resnet-50-fashion-mnist-quality-drift`).

```python
# Directory containing product and scene images
image_directory = "../images/product"
features, filenames = process_directory(image_directory, processor, model)

# Save extracted features
np.save("image_features.npy", features)
np.save("image_filenames.npy", filenames)
```

### **2. Graph Construction (`graph.py`)**

- **Process**: Constructs a K-Nearest Neighbors (KNN) graph based on cosine similarity of feature embeddings.
- **Visualization**: Uses t-SNE for dimensionality reduction and K-Means for cluster validation.
- **Output**: Graph saved as a PyTorch Geometric (PyG) data object.

```python
def create_pyg_knn_graph(features, k=5):
    # Constructs a PyG graph from KNN relationships
    # Includes bidirectional edges and similarity weights
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(features, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data
    return graph_data
```

### **3. Graph Attention Training (`GAT.py`)**

- **Process**: Trains a Graph Attention Network (GAT) for node classification.
- **Model**: Consists of two GAT layers—one for hidden node representations and another for output predictions.
- **Training**: Optimized using cross-entropy loss and the Adam optimizer.
- **Output**: Trained GAT model saved for inference.

```python
# Define GAT model with PyTorch Geometric
class GAT(torch.nn.Module):
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
```

### **4. Evaluation (`eval.py`)**

- **Process**: Generates recommendations by querying the graph using the trained GAT model.
- **Visualization**: Displays query images and their top recommended counterparts.

```python
# Load graph and GAT model
# Generate and display recommendations
graph_data_path = "graph_full.pt"
graph_data = torch.load(graph_data_path)
image_embeddings = graph_data.x.numpy()

selected_indices, selected_embeddings = select_random_images(image_embeddings, num_samples=10)
print(f"Selected indices: {selected_indices}")

recommendations = get_recommendations(trained_model, graph_data, selected_indices)
```

### **5. Metrics (`metrics.py`)**

- **Process**: Evaluates the performance of the recommendation system using metrics like precision, recall, and F1 score.
- **Output**: Quantitative measures for model performance.

```python
def evaluate_graph(feature_file, filename_file, graph_file, num_clusters=5, trained_gat_model=None):
    """
    Evaluate the graph using various metrics.
    :param feature_file: Path to the NumPy file containing features.
    :param filename_file: Path to the NumPy file containing filenames.
    :param graph_file: Path to the saved PyG graph file.
    :param num_clusters: Number of clusters for clustering-based metrics.
    :param trained_gat_model: Optional trained GAT model for additional evaluation.
    """
    features = np.load(feature_file)
    filenames = np.load(filename_file, allow_pickle=True)
    graph_data = torch.load(graph_file)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    silhouette = compute_silhouette_score(features, cluster_labels)
    davies_bouldin = compute_davies_bouldin_index(features, cluster_labels)
    density = compute_graph_density(graph_data)

    print(f"Raw Features Silhouette Score: {silhouette:.4f}")
    print(f"Raw Features Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"Graph Density: {density:.4f}")

    plot_degree_distribution(graph_data)

    with open("graph_metrics.txt", "w") as f:
        f.write(f"Raw Features Silhouette Score: {silhouette:.4f}\n")
        f.write(f"Raw Features Davies-Bouldin Index: {davies_bouldin:.4f}\n")
        f.write(f"Graph Density: {density:.4f}\n")

    if trained_gat_model is not None:
        evaluate_trained_gat_model(graph_data, trained_gat_model, num_clusters)
```

---

## Project Workflow

1. **Feature Extraction**: Extract visual embeddings from product and scene images.
2. **Graph Construction**: Build a KNN graph based on extracted features.
3. **GAT Training**: Train a GAT model to learn relationships in the graph.
4. **Evaluation**: Use the trained model to provide personalized recommendations.
5. **Metrics**: Assess model performance quantitatively.

---

## Dataset

The dataset consists of:

- **Product and Scene Images**: Images representing user preferences and fashion items.
- **Feature Annotations**: 2048-dimensional vectors reduced to 128-dimensional embeddings for efficiency.

Pre-trained ResNet-50 fine-tuned on Fashion MNIST was used for feature extraction.

---

## Results

- **Graph Clusters**: Visual clustering aligns with fashion styles and aesthetics.
- **Recommendations**: Context-aware recommendations demonstrate high relevance to query images.

---

## References

- [ResNet-50 Pre-trained Model](https://huggingface.co/arize-ai/resnet-50-fashion-mnist-quality-drift)
- [Pinterest Fashion Compatibility Dataset](https://arxiv.org/abs/2006.10792)
- [Google Mood Board Search](https://github.com/google-research/mood-board-search)

---

## Acknowledgments

We thank the CS 224W teaching staff for their support and guidance throughout the project.

---

## License

This project is licensed under the MIT License.
