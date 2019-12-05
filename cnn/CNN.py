import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class_names = []
with open("food41/meta/meta/classes.txt") as reader:
    for class_name in reader:
        class_names.append(class_name.strip())

dataset_dir = "food41/images"
model_filename = "model.h5"
batch_size = 32
num_epochs = 10
img_height = 100
img_width = 100
color_bands = 3

input_shape = (img_height, img_width, color_bands)

datagen = ImageDataGenerator(rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(img_height, img_width),
    shuffle=True,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training")

validation_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(img_height, img_width),
    shuffle=True,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation")

# Create CNN
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
    BatchNormalization(axis=-1),
    MaxPooling2D((3, 3)),
    Dropout(rate=0.25),
    
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
    BatchNormalization(axis=-1),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
    BatchNormalization(axis=-1),
    MaxPooling2D((2, 2)),
    Dropout(rate=0.25),

    Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
    BatchNormalization(axis=-1),
    Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
    BatchNormalization(axis=-1),
    MaxPooling2D((2, 2)),
    Dropout(rate=0.25),

    Flatten(),
    Dense(units=1024, activation="relu"),
    BatchNormalization(),
    Dropout(rate=0.5),
    Dense(units=len(class_names), activation="softmax")
])

# Display model summary
model.summary()

# Compile model
model.compile(optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = num_epochs)

model.save(model_filename)
