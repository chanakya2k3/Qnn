# modules/config.py

import os

# Path to the dataset directory
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')  # Adjust if your dataset is elsewhere

# Batch size for data generators
BATCH_SIZE = 32

# Number of PCA components for dimensionality reduction
PCA_COMPONENTS = 100  # Adjust based on your dataset and requirements

# Directory to save extracted features and models
FEATURES_DIR = os.path.join(os.getcwd(), 'features')
