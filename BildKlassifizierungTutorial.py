from comet_ml import Experiment
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


batch_size = 10
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/Users/philippblum/Documents/GitHub/APC/ki_project/static/images/train",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "/Users/philippblum/Documents/GitHub/APC/ki_project/static/images/validate",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# training data
print(train_ds.as_numpy_iterator().next()[0].max())

# validation data
print(val_ds.as_numpy_iterator().next()[0].max())

# normalize data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# training data
print(train_ds.as_numpy_iterator().next()[0].max())

# validation data
print(val_ds.as_numpy_iterator().next()[0].max())

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

model = keras.Sequential([
    # Data Augmentation
    preprocessing.RandomContrast(factor=0.10),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),  # Flips the image either horizontally, vertically or both.
    preprocessing.RandomRotation(factor=0.10),
    preprocessing.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),  # Randomly zooms the image 20% in and out.
    preprocessing.RandomCrop(height=img_height, width=img_width),  # Crops the image randomly.
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Translates the image by a random amount up to 10% of the image height/width.
    preprocessing.RandomRotation(factor=0.10), # Randomly rotates the image between 0 and 10 degrees.
    layers.Conv2D(32, 3, activation='relu'),

    layers.BatchNormalization(renorm=True),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.BatchNormalization(renorm=True),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.BatchNormalization(renorm=True),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax'),
])

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=1,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To log information
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
    )

def model_compile(my_model):
    experiment = Experiment(api_key="0NPgf4vYBtZjxKoE50bCNAbuL", project_name="Flower Prediction APP")
    my_model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    history_1 = my_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=50,
  callbacks=[early_stopping]
    )

model_compile(model)

model.save("ringprediction.keras") # Save the model