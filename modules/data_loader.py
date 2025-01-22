# modules/data_loader.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import DATASET_DIR, BATCH_SIZE

def create_data_generators(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE):
    """
    Creates and returns training, validation, and testing data generators.

    Parameters:
    - dataset_dir (str): Path to the dataset directory.
    - batch_size (int): Number of samples per batch.

    Returns:
    - train_generator, val_generator, test_generator: Keras ImageDataGenerator objects.
    """

    # Define paths for training, validation, and testing directories
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

    # Check if directories exist
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found at {val_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Testing directory not found at {test_dir}")

    # Initialize ImageDataGenerator for training with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,               # Normalize pixel values
        rotation_range=20,            # Random rotations
        width_shift_range=0.2,        # Random horizontal shifts
        height_shift_range=0.2,       # Random vertical shifts
        shear_range=0.2,              # Shear transformations
        zoom_range=0.2,               # Zoom operations
        horizontal_flip=True,         # Randomly flip images horizontally
        fill_mode='nearest'           # Filling strategy for new pixels
    )

    # ImageDataGenerator for validation and testing (no augmentation)
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    # Create training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),       # Resize images to 224x224
        batch_size=batch_size,
        class_mode='binary',          # Changed from 'categorical' to 'binary'
        shuffle=True
    )

    # Create validation data generator
    val_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',          # Changed from 'categorical' to 'binary'
        shuffle=False
    )

    # Create testing data generator
    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',          # Changed from 'categorical' to 'binary'
        shuffle=False
    )

    return train_generator, val_generator, test_generator
