# src/model.py

from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(48, 48, 1)):
    """
    Create and compile a Convolutional Neural Network (CNN) model for emotion classification.

    Parameters:
    - input_shape: Shape of the input images (48x48 grayscale images).

    Returns:
    - model: The compiled CNN model.
    """
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(7, activation='softmax')  # 7 emotion classes in FER-2013 dataset
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
