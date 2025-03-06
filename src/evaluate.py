# src/evaluate.py

from src.data_loader import load_preprocessed_data
from tensorflow.keras.models import load_model

# Load the test data (no augmentation)
_, _, test_generator = load_preprocessed_data(batch_size=64)

# Load the trained model
model = load_model("models/best_model.h5")

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
