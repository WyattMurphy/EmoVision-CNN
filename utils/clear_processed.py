# DESTRUCTIVE
# Clear data/processed
# Use when redoing preprocessing 

import os
import shutil

# Define paths
PROCESSED_DATA_DIR = "../data/processed"
PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
PROCESSED_VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
PROCESSED_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

def clear_directory(directory_path):

    try:
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Loop through all the items in the directory
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                
                # If file, delete it
                if os.path.isfile(file_path):
                    os.remove(file_path)
                # If directory, remove it recursively
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"Cleared all files in {directory_path}")
        else:
            print(f"Directory {directory_path} does not exist.")
    except Exception as e:
        print(f"Error clearing directory {directory_path}: {e}")

# Clear processed subdirectories
def clear_processed_data():
    print("Clearing processed data directories...")
    clear_directory(PROCESSED_TRAIN_DIR)
    clear_directory(PROCESSED_VAL_DIR)
    clear_directory(PROCESSED_TEST_DIR)

if __name__ == "__main__":
    clear_processed_data()
