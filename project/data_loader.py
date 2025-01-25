import pandas as pd
import tensorflow as tf

def load_dataset(csv_path, img_dir, img_size=(224, 224), augment=False):
    # Load CSV and create image paths
    df = pd.read_csv(csv_path)
    image_paths = [f"{img_dir}/{name}" for name in df['Image name']]
    labels = df['Risk of macular edema'].values

    # Create TensorFlow Dataset
    def parse_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0  # Normalize
        
        if augment:
            # ====== UNCOMMENT FOR AUGMENTATION ======
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            # image = tf.image.random_contrast(image, 0.9, 1.1)
        
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)