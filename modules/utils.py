# modules/utils.py

import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("pipeline.log"),
                        logging.StreamHandler()
                    ])

def save_pca_scaler(pca, scaler, features_dir):
    """
    Saves the PCA and scaler objects to the specified directory.
    """
    os.makedirs(features_dir, exist_ok=True)
    joblib.dump(pca, os.path.join(features_dir, 'pca.pkl'))
    joblib.dump(scaler, os.path.join(features_dir, 'scaler.pkl'))
    logging.info("PCA and scaler saved successfully.")

def load_pca_scaler(features_dir):
    """
    Loads the PCA and scaler objects from the specified directory.
    """
    pca = joblib.load(os.path.join(features_dir, 'pca.pkl'))
    scaler = joblib.load(os.path.join(features_dir, 'scaler.pkl'))
    logging.info("PCA and scaler loaded successfully.")
    return pca, scaler

def plot_confusion_matrix(cm, classes, title, filename, save_dir):
    """
    Plots and saves the confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    logging.info(f"Confusion matrix plot saved as {filename} in {save_dir}/")
