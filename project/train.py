import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, train_ds, val_ds, class_weights=None, epochs=10):
    # ====== UNCOMMENT FOR EARLY STOPPING ======
    # callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
    callbacks = []
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history