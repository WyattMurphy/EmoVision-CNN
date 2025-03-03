"""
data_loader.py

This module provides functions to load datasets while ensuring raw data remains untouched.
It dynamically creates a validation split from training data and applies only basic preprocessing (resizing, normalization).
"""


from tensorflow.keras.preprocessing import image_dataset_from_directory



# Image processing parameters
IMG_SIZE = (48, 48)  # Standard FER-2013 image size
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2  # 20% of training data will be used for validation

def load_data(data_dir, validation=False):
    """
    Loads and preprocesses image data while ensuring raw data remains untouched.

    Args:
        - data_dir (str): Path to the dataset directory.
        - validation (bool): Whether to return the validation split.

    Returns:
        - A `tf.data.Dataset` object.
    """
    # Check if we are dealing with training data
    is_train_data = "train" in data_dir

    # Set shuffle and seed based on whether it's training data
    shuffle = is_train_data  # Shuffle only if it's training data

    if "train" in data_dir and validation:
        print(f"Loading validation data from {data_dir} (split from training set)...")
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical", # One-hot encoding/vector encoding
            color_mode="grayscale", # Convert to grayscale 
            validation_split=VALIDATION_SPLIT,  # Create validation split dynamically
            subset="validation",
            seed=42,  # Ensure reproducibility
        )
    else:
        
        print(f"Loading data from {data_dir}...")
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            color_mode="grayscale",
            validation_split=VALIDATION_SPLIT if "train" in data_dir else None, # If data is training ensure validation split 
            subset="training" if "train" in data_dir else None, # If training handle training portion of split
            shuffle=shuffle, # shuffle if training data
            seed=42 if "train" in data_dir else None # Ensure consistent splitting only for training data
        )

    return dataset

