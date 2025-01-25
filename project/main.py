from data_loader import load_data
from model import create_model
from train import train_model

# Paths (Update these)
CSV_PATH = "C:/Users/chana/Desktop/.../Training Labels (2).csv"
IMG_DIR = "C:/Users/chana/Desktop/.../Training Set"

# Load data
train_ds, val_ds = load_data(CSV_PATH, IMG_DIR)

# Create model
model = create_model()

# Train
history = train_model(model, train_ds, val_ds, epochs=10)

# Evaluate
loss, accuracy = model.evaluate(val_ds)
print(f"\nBaseline Validation Accuracy: {accuracy*100:.2f}%")