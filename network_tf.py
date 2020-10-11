import tensorflow as tf
from tensorflow.keras import layers, models, losses
import dataset_tf
import numpy as np


net = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(40, 40, 1)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1024)
])

net.summary()
net.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

train_images, train_labels = dataset_tf.generate_dataset(20000)
validation_set = dataset_tf.generate_dataset(10)

net.fit(train_images, train_labels, epochs=50, validation_data=validation_set)
"""
batch_size = 20

for i in range(1000):
    print('Iteration %i' % i)

    selection = np.random.randint(0, 1000, batch_size)

    s_images = train_images[selection]
    s_labels = train_labels[selection]

    net.fit(s_images, s_labels, epochs=5, validation_data=validation_set)
"""
