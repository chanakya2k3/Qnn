# modules/feature_extractor.py

import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tqdm import tqdm
from .config import FEATURES_DIR
from .utils import save_pca_scaler, load_pca_scaler

class FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 3), pooling='avg'):
        """
        Initializes the FeatureExtractor with a pre-trained CNN model.

        Parameters:
        - input_shape (tuple): Shape of the input images.
        - pooling (str): Pooling mode for feature extraction.
        """
        # Load the ResNet50 model without the top classification layers
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling=pooling)
        self.model = base_model
        print("Pre-trained ResNet50 model loaded for feature extraction.")

    def extract_features(self, data_generator, dataset_type='train'):
        """
        Extracts features from images using the pre-trained CNN model.

        Parameters:
        - data_generator: Keras ImageDataGenerator object.
        - dataset_type (str): Type of dataset ('train', 'validation', 'test').

        Returns:
        - features (numpy.ndarray): Extracted feature vectors.
        - labels (numpy.ndarray): Corresponding labels.
        """

        # Number of samples and feature dimensions
        num_samples = data_generator.samples
        feature_dim = self.model.output_shape[-1]
        
        # Initialize arrays to hold features and labels
        features = np.zeros((num_samples, feature_dim))
        labels = np.zeros((num_samples,), dtype=int)  # 1D array for binary labels
        
        # Index to keep track of the current position in the arrays
        i = 0
        
        print(f"Extracting features for {dataset_type} dataset...")

        # Iterate over the data generator
        for inputs_batch, labels_batch in tqdm(data_generator, total=np.ceil(num_samples / data_generator.batch_size)):
            # Use the model to predict features
            features_batch = self.model.predict(inputs_batch)
            
            # Assign features and labels to the arrays
            batch_size_actual = inputs_batch.shape[0]
            features[i * data_generator.batch_size : i * data_generator.batch_size + batch_size_actual] = features_batch
            labels[i * data_generator.batch_size : i * data_generator.batch_size + batch_size_actual] = labels_batch.astype(int)
            
            i += 1
            if i * data_generator.batch_size >= num_samples:
                break  # Exit loop once all samples are processed
        
        print(f"Features extracted for {dataset_type} dataset.")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return features, labels
