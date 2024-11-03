# VibeRec
**CS 224W: Vibe Recommendation System**

VibeRec is a project aimed at recommending fashion styles based on visual features extracted from images. The pipeline begins with `feature.py`, which extracts feature embeddings from fashion images. These embeddings are then processed through a **Graph Convolutional Network (GCN)** and potentially a **Graph Attention Network (GAT)** to analyze trends and personalize recommendations based on recent visual styles in fashion.

## Overview of `feature.py`: Feature Extraction

The `feature.py` script is the foundation of the VibeRec pipeline, performing essential steps to prepare image data for graph-based analysis.

### Steps in `feature.py`

1. **Image Download and Preprocessing**  
   - `feature.py` downloads fashion images (products and scenes) based on URLs in the dataset.
   - Images are organized into separate folders (`images/product` and `images/scene`) for better management.
   - The script processes each image by resizing, normalizing, and converting it to a tensor format compatible with a pre-trained ResNet-50 model.

2. **Feature Vector Extraction**  
   - Using a pre-trained **ResNet-50** model, `feature.py` extracts high-level features from each image.
   - The final layer of ResNet-50 is removed to obtain a 2048-dimensional feature vector from each image, capturing visual characteristics like texture, color, and shape.
   - A two-layer feedforward neural network further reduces the feature vector to a compact 128-dimensional embedding, optimized for style similarity.

### Next Steps: Graph Neural Network (GNN) Analysis

The extracted embeddings serve as input to graph-based models, which can capture relationships between fashion items.

1. **Graph Convolutional Network (GCN)**
   - A **GCN** can be used to model connections between different fashion items based on their embeddings.
   - Each node in the graph represents a fashion item, and edges represent similarity based on features (e.g., similar color, style, or texture).
   - The GCN learns to propagate information across nodes, capturing underlying patterns in fashion trends.

2. **Graph Attention Network (GAT)**
   - A **GAT** adds an attention mechanism to focus on specific relationships between items.
   - For example, the GAT might prioritize connections between trending items or styles that are frequently paired together.
   - By using attention, the model can adapt to recent fashion trends, dynamically emphasizing popular styles.

3. **Analysis of Feature Embeddings**
   - Feature embeddings are analyzed by comparing items' similarity and grouping them into clusters.
   - Clustering can reveal distinct fashion trends or styles, which can be visualized to gain insights into popular aesthetics.
   - These embeddings are transformed within the GNN to create a more meaningful representation, enabling accurate recommendations based on recent trends.

## Running `feature.py` and Requirements

### Dependencies and `requirements.txt`
To ensure `feature.py` runs smoothly, install the necessary dependencies. The main packages are listed in `requirements.txt`, which can be installed as follows:

```bash
pip install -r requirements.txt
```

### Running `feature.py` in the Background

You can run `feature.py` in the background to allow it to continue processing even if you close the terminal.

#### Using `nohup` to Run in the Background

You can use `nohup` to start the script in the background, which redirects the output to a log file:

```bash
nohup python feature.py > output.log 2>&1 &
```

#### Checking Progress with nohup
To monitor the progress of the script while it runs, view the output.log file in real-time:

```bash
tail -f output.log
```