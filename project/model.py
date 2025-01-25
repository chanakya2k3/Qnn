import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3), num_classes=3):
    # Basic CNN (KAN-inspired simplified version)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # ====== IMPROVEMENTS TO UNCOMMENT LATER ======
    # 1. Add Dropout layers
    # 2. Add L2 Regularization
    # 3. Add BatchNorm layers
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model