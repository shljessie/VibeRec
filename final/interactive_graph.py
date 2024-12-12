import os
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px

# Step 1: Load Features and Filenames
def load_features_and_filenames(feature_file, filename_file):
    """
    Load extracted features and filenames from files.
    :param feature_file: Path to the NumPy file containing features.
    :param filename_file: Path to the NumPy file containing filenames.
    :return: Tuple of features (NumPy array) and filenames (list).
    """
    features = np.load(feature_file)
    filenames = np.load(filename_file, allow_pickle=True)
    return features, filenames

# Step 2: Reduce Features to 2D
def reduce_to_2d(features):
    """
    Reduce high-dimensional features to 2D using t-SNE.
    :param features: NumPy array of feature embeddings.
    :return: 2D array of reduced features.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

# Step 3: Create Interactive Visualization
def visualize_with_hover(reduced_features, filenames, image_directory):
    """
    Create an interactive scatter plot where hovering over a point shows the image.
    :param reduced_features: 2D array of reduced features.
    :param filenames: List of image filenames.
    :param image_directory: Directory where the images are stored.
    """
    # Prepare data for Plotly
    data = pd.DataFrame({
        "x": reduced_features[:, 0],
        "y": reduced_features[:, 1],
        "image": [os.path.join(image_directory, fname) for fname in filenames]
    })

    # Embed images in hover tooltips
    data["hover_image"] = data["image"].apply(
        lambda path: f"<img src='{path}' style='max-width:200px;'>"
    )

    # Create scatter plot with hover image
    fig = px.scatter(
        data,
        x="x",
        y="y",
        title="Interactive Visualization with Hover Images"
    )

    # Add custom hover templates to show images
    fig.update_traces(
        hovertemplate="<br>%{customdata}<extra></extra>",
        customdata=data["hover_image"].values
    )

    # Display the plot
    fig.show()

if __name__ == '__main__':
    # File paths
    feature_file = "image_features.npy"
    filename_file = "image_filenames.npy"
    image_directory = "../images/product"

    # Step 1: Load Features and Filenames
    print("Loading features and filenames...")
    features, filenames = load_features_and_filenames(feature_file, filename_file)

    # Step 2: Reduce Features to 2D
    print("Reducing features to 2D...")
    reduced_features = reduce_to_2d(features)

    # Step 3: Create Interactive Visualization
    print("Creating interactive visualization...")
    visualize_with_hover(reduced_features, filenames, image_directory)
