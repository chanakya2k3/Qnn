import pandas as pd
import tensorflow as tf
import os

def load_dataset(csv_path, img_dir, img_size=(224, 224), augment=False):
    # Load CSV and clean filenames
    df = pd.read_csv(csv_path)
    df['Image name'] = df['Image name'].str.strip() + '.jpg'  # Ensure .jpg extension
    
    # Construct proper Windows paths
    image_paths = [os.path.join(img_dir, fname) for fname in df['Image name']]
    
    # Verify first 5 paths exist
    for path in image_paths[:5]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
    
    labels = df['dme'].values

    def parse_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0  # Normalize
        
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)