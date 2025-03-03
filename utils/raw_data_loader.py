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
SEED = 42

def load_raw_data(data_dir, validation=False):
    """
    Loads raw image data from the raw directory and applies basic resizing.
    
    Args:
        data_dir (str): Path to the raw dataset directory ("raw/train" or "raw/test").
        validation (bool): If True, loads the validation split from training data.
    
    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """
    # Check if we are dealing with training data
    is_train_data = "train" in data_dir

    # Set shuffle and seed based on whether it's training data
    shuffle = is_train_data  # Shuffle only if it's training data

    if is_train_data and validation:
        print(f"Loading validation data from {data_dir} (split from training set)...")
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical", # One-hot encoding/vector encoding
            color_mode="grayscale", # Convert to grayscale 
            validation_split=VALIDATION_SPLIT,  # Create validation split dynamically
            subset="validation",
            seed=SEED,  # Ensure reproducibility
        )
    else:
        
        print(f"Loading data from {data_dir}...")
        dataset = image_dataset_from_directory(
            data_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            color_mode="grayscale",
            validation_split=VALIDATION_SPLIT if is_train_data else None, # If data is training ensure validation split 
            subset="training" if is_train_data else None, # If training handle training portion of split
            shuffle=shuffle, # shuffle if training data
            seed=SEED if is_train_data else None # Ensure consistent splitting only for training data
        )

    return dataset

