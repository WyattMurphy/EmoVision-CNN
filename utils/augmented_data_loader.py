import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
PROCESSED_DATA_DIR = "../data/processed"

# def load_preprocessed_data_augmented(batch_size=64, target_size=(48, 48), color_mode='grayscale'):
#     """
#     Loads preprocessed data (already normalized and grayscale) from the processed directories.
#     Applies on-the-fly augmentation for the training dataset.

#     Parameters:
#     - batch_size: The batch size for loading data.
#     - target_size: The target size of images to resize to (default is 48x48, which matches FER-2013).
#     - color_mode: Set to 'grayscale' as the FER-2013 dataset is grayscale.

#     Returns:
#     - train_generator: Generator for training data (with augmentation).
#     - val_generator: Generator for validation data (no augmentation).
#     - test_generator: Generator for test data (no augmentation).
#     """
    
#     # the directories for preprocessed data
#     train_dir = "../data/processed/train"
#     val_dir = "../data/processed/val"
#     test_dir = "../data/processed/test"
    
#     # check the directories exist
#     if not (os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)):
#         raise ValueError(f"One or more directories do not exist: {train_dir}, {val_dir}, {test_dir}")
    
#     # data augmentation for training set
#     train_datagen = ImageDataGenerator(
#         rotation_range=30,        # randomly rotate images by up to 30 degrees
#         width_shift_range=0.12,    # shift width by up to 12%
#         height_shift_range=0.12,   # shift height by up to 12%
#         zoom_range=0.20,           # random zoom by up to 20%
#         horizontal_flip=True,     # flip images horizontally
#         fill_mode='nearest'       # fill mode for newly created pixels
#     )

#     # no augmentation for validation and test data
#     val_test_datagen = ImageDataGenerator()

#     # load training data with augmentation
#     train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=target_size,
#         batch_size=batch_size,
#         class_mode='categorical',  
#         color_mode=color_mode,
#         shuffle=True  # shuffle training data for better generalization
#     )

#     # load validation data (no augmentation)
#     val_generator = val_test_datagen.flow_from_directory(
#         val_dir,
#         target_size=target_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         color_mode=color_mode,
#         shuffle=False  # no shuffling for validation/test sets
#     )

#     # load test data (no augmentation)
#     test_generator = val_test_datagen.flow_from_directory(
#         test_dir,
#         target_size=target_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         color_mode=color_mode,
#         shuffle=False
#     )

#     # return the generators for use in model training and evaluation
#     return train_generator, val_generator, test_generator


def augment_and_load_train_gen(batch_size=64, target_size=(48, 48), color_mode='grayscale'):
    # establish path to processed training data. 
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
        # check the directories exist
    if not (os.path.exists(train_dir)):
        raise ValueError(f"Train directory does not exist: {train_dir}")
    
    # define data augmentation for traiing set
    train_datagen = ImageDataGenerator(
        rotation_range=30,        # randomly rotate images by up to 30 degrees
        width_shift_range=0.12,    # shift width by up to 12%
        height_shift_range=0.12,   # shift height by up to 12%
        zoom_range=0.20,           # random zoom by up to 20%
        horizontal_flip=True,     # flip images horizontally
        fill_mode='nearest'       # fill mode for newly created pixels
    )

    # load training data with augmentation
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  
        color_mode=color_mode,
        shuffle=True  # shuffle training data for better generalization
    )

    return train_generator

def load_test_val_gen(batch_size=64, target_size=(48, 48), color_mode='grayscale'):
    # load preprocessed data
    val_dir = os.path.join(PROCESSED_DATA_DIR, "val")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test")
    
    # check the directories exist
    if not (os.path.exists(val_dir) & os.path.exists(test_dir)):
        raise ValueError(f"One of the directories does not exist: {test_dir}, {val_dir}")

     # no augmentation for validation and test data
    val_test_datagen = ImageDataGenerator()
   
    # load validation data (no augmentation)
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False  # no shuffling for validation/test sets
    )

    # load test data (no augmentation)
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False
    )

    return val_dir, test_dir