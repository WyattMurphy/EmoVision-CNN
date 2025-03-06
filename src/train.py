# src/train.py

from utils.augmented_data_loader import load_preprocessed_data_augmented
from model import create_cnn_model
import os
from tensorflow.keras.callbacks import TensorBoard

# set up TensorBoard log directory
log_dir = "../logs/tensorboard_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# initialize TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# load the preprocessed data with augmentation for training
train_generator, val_generator, _ = load_preprocessed_data_augmented(batch_size=64)

# create the CNN model
model = create_cnn_model()

# train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# save the trained model
model.save("../models/best_model.h5")

# evaluation

