import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # ====== UNCOMMENT FOR REGULARIZATION ======
        # layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model