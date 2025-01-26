import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_dataset
from model import create_model
from train import train_model

# PATHS (REPLACE WITH YOUR ACTUAL PATHS)
train_csv = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels (2).csv"
test_csv = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels (2).csv"
train_img_dir = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\1. Original Images\a. Training Set"
test_img_dir = r"C:\Users\chana\Desktop\datasets\B. Disease Grading\B. Disease Grading\1. Original Images\b. Testing Set"

# LOAD AND SPLIT TRAINING DATA
train_df = pd.read_csv(train_csv)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['dme'])

# CREATE DATASETS
train_ds = load_dataset(train_csv, train_img_dir, augment=False)  # UNCOMMENT `augment=True` LATER
val_ds = load_dataset(train_csv, train_img_dir)  # Use same CSV but different split
test_ds = load_dataset(test_csv, test_img_dir)

# COMPUTE CLASS WEIGHTS (UNCOMMENT TO ADDRESS IMBALANCE)
# class_counts = train_df['dme'].value_counts()
# class_weights = {cls: len(train_df)/(3 * count) for cls, count in class_counts.items()}
class_weights = None

# TRAIN MODEL
model = create_model()
history = train_model(model, train_ds, val_ds, class_weights, epochs=10)

# EVALUATE
loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")