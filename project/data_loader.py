import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(csv_path, img_dir, img_size=(224, 224), test_size=0.2):
    # Load CSV labels
    df = pd.read_csv(csv_path)
    df['Image name'] = df['Image name'] + '.jpg'  # Add .jpg extension
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['Risk of macular edema'])
    
    # Basic DataLoader (No augmentation)
    def create_dataset(df):
        def parse_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, img_size)
            image = tf.cast(image, tf.float32) / 255.0  # Normalize
            return image
        
        image_paths = [f"{img_dir}/{name}" for name in df['Image name']]
        labels = df['Risk of macular edema'].values
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(lambda x, y: (parse_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds
    
    train_ds = create_dataset(train_df).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = create_dataset(val_df).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# ====== IMPROVEMENTS TO UNCOMMENT LATER ======
# 1. Add class weights for imbalance
# 2. Add data augmentation
# 3. Add SMOTE oversampling