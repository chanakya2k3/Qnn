# modules/classifier.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Example classical classifier
from sklearn.metrics import classification_report, confusion_matrix
from .config import PCA_COMPONENTS, FEATURES_DIR
from .utils import save_pca_scaler, load_pca_scaler, plot_confusion_matrix

class ClassicalClassifier:
    def __init__(self, pca_components=PCA_COMPONENTS):
        """
        Initializes the ClassicalClassifier with scaling, PCA, and the classifier.

        Parameters:
        - pca_components (int): Number of PCA components.
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        self.classifier = SVC(kernel='linear')  # Example: Linear SVM

    def prepare_data(self, X_train, X_val, X_test):
        """
        Scales and applies PCA to the feature data.

        Parameters:
        - X_train (numpy.ndarray): Training features.
        - X_val (numpy.ndarray): Validation features.
        - X_test (numpy.ndarray): Testing features.

        Returns:
        - X_train_pca, X_val_pca, X_test_pca (numpy.ndarray): PCA-transformed features.
        """
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Save PCA and scaler
        save_pca_scaler(self.pca, self.scaler, FEATURES_DIR)
        
        return X_train_pca, X_val_pca, X_test_pca

    def train(self, X_train, y_train):
        """
        Trains the classifier on the training data.

        Parameters:
        - X_train (numpy.ndarray): PCA-transformed training features.
        - y_train (numpy.ndarray): 1D training labels.
        """
        self.classifier.fit(X_train, y_train)
        print("Classifier trained successfully.")
        return self.classifier

    def evaluate(self, X_test, y_test):
        """
        Evaluates the classifier on the testing data.

        Parameters:
        - X_test (numpy.ndarray): PCA-transformed testing features.
        - y_test (numpy.ndarray): 1D testing labels.
        """
        y_pred = self.classifier.predict(X_test)
        accuracy = self.classifier.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Classifier Accuracy: {accuracy * 100:.2f}%\n")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, ['class0', 'class1'], 'Confusion Matrix', 'confusion_matrix.png', FEATURES_DIR)
