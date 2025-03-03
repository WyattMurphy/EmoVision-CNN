"""AI GENERATED TEST FOR IMPORTING TENSORFLOW"""

import tensorflow as tf

# 1. Check TensorFlow version
def test_tensorflow_version():
    print("TensorFlow version:", tf.__version__)
    assert tf.__version__ >= '2.0', "TensorFlow version is lower than 2.0. Please update."

# 2. Verify TensorFlow Import
def test_tensorflow_import():
    try:
        import tensorflow as tf
        print("TensorFlow import successful")
    except ImportError:
        print("TensorFlow import failed")

# 3. Test Keras Import (for image_dataset_from_directory)
def test_keras_import():
    try:
        from tensorflow.keras.preprocessing import image_dataset_from_directory
        print("image_dataset_from_directory import successful")
    except ImportError:
        print("Failed to import image_dataset_from_directory")
    
    # Check the contents of the module to see if the function exists
    try:
        from tensorflow.keras import preprocessing
        print("Contents of preprocessing module:", dir(preprocessing))
    except ImportError:
        print("Failed to import tensorflow.keras.preprocessing")

# 4. Test basic Keras functionality (creating a simple model)
def test_keras_functionality():
    try:
        from tensorflow.keras.models import Sequential
        model = Sequential()
        print("Keras model created:", model)
    except Exception as e:
        print("Error creating Keras model:", e)

# 5. Test eager execution (for TensorFlow 2.x)
def test_eager_execution():
    print("Eager execution enabled:", tf.executing_eagerly())
    if not tf.executing_eagerly():
        print("Eager execution is not enabled. Enabling it manually.")
        tf.config.run_functions_eagerly(True)

# 6. Test loading a sample dataset using image_dataset_from_directory
def test_image_dataset_from_directory():
    try:
        from tensorflow.keras.preprocessing import image_dataset_from_directory
        dataset = image_dataset_from_directory(
            'path_to_your_images_directory',  # Replace with actual path
            image_size=(256, 256),
            batch_size=32
        )
        print("Dataset loaded successfully:", dataset)
    except Exception as e:
        print("Error loading dataset:", e)

# 7. Check if TensorFlow detects GPUs (if applicable)
def test_gpu_detection():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == "__main__":
    test_tensorflow_version()
    test_tensorflow_import()
    test_keras_import()
    test_keras_functionality()
    test_eager_execution()
    test_image_dataset_from_directory()
    test_gpu_detection()
