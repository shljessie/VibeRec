# Personalized Complete the Look

This project explores a novel approach to personalized fashion recommendation systems by leveraging user-curated visual collections (e.g., moodboards) and graph-based learning techniques. The system combines advanced feature extraction, graph construction, and Graph Attention Networks (GAT) to provide contextually relevant and aesthetically aligned recommendations.

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
    ...
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
    ...
# Train and save the model
...
```

### **4. Evaluation (`eval.py`)**

- **Process**: Generates recommendations by querying the graph using the trained GAT model.
- **Visualization**: Displays query images and their top recommended counterparts.

```python
# Load graph and GAT model
...
# Generate and display recommendations
...
```

### **5. Metrics (`metrics.py`)**

- **Process**: Evaluates the performance of the recommendation system using metrics like precision, recall, and F1 score.
- **Output**: Quantitative measures for model performance.

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

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. Ensure PyTorch and PyTorch Geometric are installed for your system:

```bash
pip install torch torchvision torch-geometric
```

---

## Usage

### Feature Extraction

```bash
python feature.py
```

### Graph Construction

```bash
python graph.py
```

### GAT Training

```bash
python GAT.py
```

### Evaluation

```bash
python eval.py
```

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
