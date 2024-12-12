import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import matplotlib.pyplot as plt

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

def find_similar_images(features, filenames, query_index, k=5):
    """
    Find k-nearest neighbors for a given query image.
    :param features: NumPy array of feature embeddings.
    :param filenames: List of image filenames.
    :param query_index: Index of the query image in the dataset.
    :param k: Number of similar images to retrieve.
    :return: List of indices and filenames of similar images.
    """
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    knn.fit(features)
    distances, indices = knn.kneighbors(features[query_index].reshape(1, -1))
    
    similar_indices = indices[0][1:]
    similar_filenames = [filenames[i] for i in similar_indices]
    return similar_indices, similar_filenames

def visualize_similar_images(query_image_path, similar_image_paths, query_index):
    """
    Visualize the query image and its similar images.
    :param query_image_path: Path to the query image.
    :param similar_image_paths: List of paths to similar images.
    :param query_index: Index of the query image.
    """
    images = [Image.open(query_image_path)] + [Image.open(img) for img in similar_image_paths]
    titles = ["Query Image"] + [f"Similar Image {i+1}" for i in range(len(similar_image_paths))]

    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"similar_images_query_{query_index}.png")
    plt.show()

if __name__ == '__main__':
    feature_file = "image_features.npy"
    filename_file = "image_filenames.npy"
    image_directory = "../images/product"

    print("Loading features and filenames...")
    features, filenames = load_features_and_filenames(feature_file, filename_file)

    query_indices = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    k = 5 

    for i, query_index in enumerate(query_indices):
        print(f"\nProcessing query image {i + 1}/{len(query_indices)} (Index: {query_index})...")

        query_image_path = os.path.join(image_directory, filenames[query_index])

        similar_indices, similar_filenames = find_similar_images(features, filenames, query_index, k=k)
        similar_image_paths = [os.path.join(image_directory, fname) for fname in similar_filenames]

        print(f"Visualizing query image {i + 1} and its similar images...")
        visualize_similar_images(query_image_path, similar_image_paths, query_index)
