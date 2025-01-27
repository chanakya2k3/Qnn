import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_dataset
from model import create_model
from train import train_model

# Paths
train_csv = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels (2).csv"
test_csv = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels (2).csv"
train_img_dir = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\1. Original Images\a. Training Set"
test_img_dir = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\1. Original Images\b. Testing Set"

# Load and split data
train_df = pd.read_csv(train_csv)
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['dme'],
    random_state=42
)

# Create datasets
train_ds = load_dataset(train_df, train_img_dir, augment=True)  # Augmentation ON
val_ds = load_dataset(val_df, train_img_dir)  # Use validation split
test_ds = load_dataset(pd.read_csv(test_csv), test_img_dir)

# Class weights
class_counts = train_df['dme'].value_counts()
class_weights = {cls: (1/class_counts[cls])*(len(train_df)/2.0) for cls in class_counts.index}

# Build and train
model = create_model()
history = train_model(model, train_ds, val_ds, class_weights, epochs=50)

# Evaluate
loss, accuracy = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")