from comet_ml import Experiment
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/Users/philippblum/Desktop/coding/FlowerPredictionAPP/flower_photos/train",
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "/Users/philippblum/Desktop/coding/FlowerPredictionAPP/flower_photos/validate",
  subset="validation",
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
    #preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.10),
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

def model_compile(my_model):
    experiment = Experiment(api_key="0NPgf4vYBtZjxKoE50bCNAbuL", project_name="Flower Prediction APP")
    my_model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history_1 = my_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=4
)

model_compile(model)

model.save("flowerprediction.h5")