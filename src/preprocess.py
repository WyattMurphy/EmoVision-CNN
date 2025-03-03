"""
preprocess.py

Run Once

Script responsible for preprocessing the dataset
1. The dataset is loaded correctly from `data/raw/`.
2. The training data is split dynamically into training and validation.
3. The processed dataset is optionally saved to `data/processed/`.
4. No data augmentation is applied at this stage (only resizing and grayscale conversion).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from utils import data_loader

# Define paths
RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"

TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")
TEST_DIR = os.path.join(RAW_DATA_DIR, "test")

PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
PROCESSED_VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
PROCESSED_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

# Ensure processed directories exist
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_VAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

def preprocess_and_save():
    """
    Loads and processes the dataset. Train data is split into train/val 
    and copies test data to the processed directory.
    """
    print("Starting preprocessing...")

    # Load training dataset (automatically splits into training & validation)
    train_dataset = data_loader.load_data(TRAIN_DIR, validation=False)
    val_dataset = data_loader.load_data(TRAIN_DIR, validation=True)

    # Load test dataset (no augmentation, just resizing and grayscale conversion)
    test_dataset = data_loader.load_data(TEST_DIR, validation=False)

    print("\n" * 15)
    # Print the structure of the train dataset (element_spec describes the data type and shape of the dataset)
    print("Train dataset element_spec:", train_dataset.element_spec)

    # Print the structure of the validation dataset (element_spec describes the data type and shape of the dataset)
    print("Validation dataset element_spec:", val_dataset.element_spec)

    # Print the structure of the test dataset (element_spec describes the data type and shape of the dataset)
    print("Test dataset element_spec:", test_dataset.element_spec)
    print("\n" * 15)

    # Save datasets to processed directory 
    print("Saving processed datasets...")
    train_dataset.save(PROCESSED_TRAIN_DIR)
    val_dataset.save(PROCESSED_VAL_DIR)
    test_dataset.save(PROCESSED_TEST_DIR)

    print("Preprocessing complete. Data saved to `data/processed/`.")

if __name__ == "__main__":
    preprocess_and_save()
