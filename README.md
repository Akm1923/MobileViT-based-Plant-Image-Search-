# MobileViT-based Plant Image Search

This project implements a plant image search engine using a lightweight, MobileViT-inspired model called FastLightCLIP. The system takes a plant image as a query and retrieves the most visually similar images from a large dataset.

## Dataset

The project uses the [Plants Type Datasets](https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets) from Kaggle. This dataset contains thousands of images of various plant species, which are used to build the image feature index for the search engine.

## Model: FastLightCLIP

To enable efficient image retrieval on resource-constrained devices, this project introduces **FastLightCLIP**, a lightweight multimodal model inspired by the architecture of CLIP and MobileViT. It consists of two main components:

1.  **Vision Encoder**: A lightweight vision transformer based on MobileNetV3-Small, which is used to extract compact and meaningful feature embeddings from images.
2.  **Text Encoder**: A distilled BERT-like model that encodes textual descriptions. While the text encoder is defined, this project primarily focuses on the image search functionality.

The vision encoder processes images from the dataset to generate a 256-dimensional feature vector for each image. These features are L2-normalized to allow for efficient similarity calculation using dot products.

## How It Works

1.  **Feature Extraction**: The `plant_image_search.ipynb` notebook begins by downloading the plant dataset. It then iterates through all the images, using the pre-trained `VisionEncoder` to extract a feature vector for each one.
2.  **Indexing**: The extracted image features are stored in a NumPy file named `fastlightclip_image_features.npy`, and the corresponding image paths are saved in `fastlightclip_image_paths.npy`. Together, these files act as a simple searchable index.
3.  **Image Search**: When a user provides a query image, its features are extracted using the same `VisionEncoder`. The system then computes the cosine similarity (via dot product) between the query feature and all the indexed features.
4.  **Results**: The images with the highest similarity scores are retrieved and displayed as the top search results.

## How to Run

1.  **Set up the Environment**: Ensure you have Python, Jupyter Notebook, and the required libraries (`torch`, `torchvision`, `transformers`, `numpy`, `matplotlib`, `kaggle`) installed.
2.  **Kaggle API Key**: To download the dataset, you'll need a Kaggle account and your API key (`kaggle.json`). Place this file in the appropriate directory as instructed in the notebook.
3.  **Run the Notebook**: Open and run the `plant_image_search.ipynb` notebook. The notebook will:
    *   Download and extract the dataset.
    *   Build and cache the feature index files (`.npy`).
    *   Allow you to specify a path to a query image and visualize the search results.

## Example Results

Here are some examples of the image search in action:

**Query 1: A healthy, leafy plant**

The system successfully retrieves other images of similar-looking plants from the dataset.

**Query 2: A plant with distinct features**

Even for more unique plants, the model is able to find visually relevant matches.

**Query 3: An out-of-dataset image**

When tested with an image not present in the dataset, the model still attempts to find the closest available matches based on visual patterns.

This demonstrates the model's ability to generalize and find relevant images even for novel inputs.
