""" 
    AI Generated Test To Validate Preprocessing
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Define paths
PROCESSED_TRAIN_DIR = "../data/processed/train"
PROCESSED_VAL_DIR = "../data/processed/val"
PROCESSED_TEST_DIR = "../data/processed/test"

def test_processed_data():

    # Load the datasets
    train_dataset = tf.data.Dataset.load(PROCESSED_TRAIN_DIR)
    val_dataset = tf.data.Dataset.load(PROCESSED_VAL_DIR)
    test_dataset = tf.data.Dataset.load(PROCESSED_TEST_DIR)

    # Check the shape of the datasets (first batch)
    for dataset, name in zip([train_dataset, val_dataset, test_dataset],
                             ['Train', 'Validation', 'Test']):
        print(f"\n{name} dataset preview:")
        
        for images, labels in dataset.take(1):  # Inspect the first batch
            print(f"  - Image shape: {images.shape}")  # Should be (batch_size, 48, 48, 1)
            print(f"  - Label shape: {labels.shape}")  # Should be (batch_size, 7)

            # Verify the image size and grayscale channel
            assert images.shape[1:] == (48, 48, 1), f"Image shape mismatch in {name} dataset"
            assert labels.shape[1] == 7, f"Label shape mismatch in {name} dataset"

            # Display a sample image and label
            plt.imshow(images[0].numpy().squeeze(), cmap='gray')
            plt.title(f"Sample Label (one-hot): {labels[0].numpy()}")
            plt.show()

            # Verify the labels are one-hot encoded (sum should be 1)
            assert np.sum(labels[0].numpy()) == 1, f"Label not one-hot encoded in {name} dataset"

def main():
    """
    Run the test for the processed datasets.
    """
    test_processed_data()
    print("\nProcessed datasets passed the test!")

if __name__ == "__main__":
    main()

