import tensorflow as tf  # <-- Add this line
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Now works
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model