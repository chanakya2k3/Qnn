import tensorflow as tf

def train_model(model, train_ds, val_ds, epochs=10):
    # Basic training (No callbacks)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    # ====== IMPROVEMENTS TO UNCOMMENT LATER ======
    # 1. Add EarlyStopping
    # 2. Add ModelCheckpoint
    # 3. Add class weights
    
    return history