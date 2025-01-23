import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set dataset path
base_dir = '/content/drive/My Drive/pjt2_dataset/Diagnosis of Diabetic Retinopathy'

# Create paths to dataset splits
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Data preprocessing and generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['No_DR', 'DR']
)

validation_generator = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['No_DR', 'DR']
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['No_DR', 'DR']
)

# Build CNN model (same architecture as before)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest accuracy: {test_acc:.2f}')

# Feature extraction
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-4].output  # Flatten layer
)

def extract_features(generator, sample_count):
    features = np.zeros(shape=(sample_count, model.layers[-4].output_shape[1]))
    labels = np.zeros(shape=(sample_count))
    
    generator.reset()
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = feature_extractor.predict(inputs_batch)
        batch_size = inputs_batch.shape[0]
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# Extract features
train_features, train_labels = extract_features(train_generator, train_generator.samples)
val_features, val_labels = extract_features(validation_generator, validation_generator.samples)
test_features, test_labels = extract_features(test_generator, test_generator.samples)

# Save features if needed
np.save(os.path.join(base_dir, 'train_features.npy'), train_features)
np.save(os.path.join(base_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(base_dir, 'val_features.npy'), val_features)
np.save(os.path.join(base_dir, 'val_labels.npy'), val_labels)
np.save(os.path.join(base_dir, 'test_features.npy'), test_features)
np.save(os.path.join(base_dir, 'test_labels.npy'), test_labels)