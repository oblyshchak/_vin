#!/usr/bin/env python3

import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# load and split the data to train, validation and test subset
(train_ds, val_ds, test_ds), metadata = tfds.load('emnist/balanced',
                                                 split=['train[:75%]', 'train[75%:90%]', 'train[90%:]'],
                                                 with_info=True,
                                                 as_supervised=True)

# get label, mapping the data
label_names = metadata.features['label'].names
label_indices = metadata.features['label'].int2str
label_mapping = {index: name for index, name in enumerate(label_names)}

# data overview
train_images = tf.data.experimental.cardinality(train_ds).numpy()
print("Number of train images:", train_images)
test_images = tf.data.experimental.cardinality(test_ds).numpy()
print("Number of test images:", test_images)
val_images = tf.data.experimental.cardinality(val_ds).numpy()
print("Number of validation images:", val_images)

# function to filter data, save data which represent lebels from 0 to 35 --> 0-9, A-Z, upper only, VIN contains only upper letters
def filter_lowercase(image, label):
    # Keep only the labels from 0 to 35
    return tf.math.less(label, 36)

train_ds = train_ds.filter(filter_lowercase)
val_ds = val_ds.filter(filter_lowercase)
test_ds = test_ds.filter(filter_lowercase)

# function to preprocess images, original dataset has images, which are rotated and flipped 
def preprocess(image, label):
    # rotate by -90 degrees
    image = tf.image.rot90(image, k=3)
    # flip the image horizontally
    image = tf.image.flip_left_right(image)
    # convert to float type and scaled to between 0 and 1
    image = tf.cast(image, tf.float32) / 255.0
    # resize images for having same dimension, shape
    image = tf.image.resize(image, (28, 28))
    return image, label

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# # visualizate preprocessed images with original labels
# for image, label in train_ds.take(5):
#     plt.imshow(image.numpy(), cmap='gray')
#     plt.axis('off')
#     plt.show()
#     print("Label:", label.numpy())

# groupind data samples to improve efficience
batch_size = 1000
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# create CNN model 
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(36, activation = 'softmax'))

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
epochs = 30
model.summary()

# fit the model
model.fit(train_ds, validation_data=val_ds, epochs = epochs)

loss, accuracy = model.evaluate(test_ds)
print(f"Accuracy test set: {accuracy}")

# save binary model
model.save('model.h5')
