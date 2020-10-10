import tensorflow as tf
from tensorflow.keras import layers, models


net = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 1)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(4*8)
])

net.summary()
