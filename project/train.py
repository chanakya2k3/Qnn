from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, train_ds, val_ds, class_weights=None, epochs=30):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    return history