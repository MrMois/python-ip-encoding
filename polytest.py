import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, losses


img_size = 50
min_off  = 5

def generate_training_pair():

    img = np.zeros((img_size, img_size))
    off = np.random.randint(min_off, int(img_size/2-2*min_off), 8)

    corners = np.array([
        [off[0],            off[1]          ],
        [img_size-off[3],   off[2]          ],
        [img_size-off[4],   img_size-off[5] ],
        [off[6],            img_size-off[7] ],
    ], dtype=np.int)

    cv2.fillPoly(img, pts=[corners], color=(255, 255, 255))

    return img / 255.0, corners.reshape(8) / img_size

def generate_dataset(size):

    inputs = np.zeros((size, img_size, img_size))
    labels = np.zeros((size, 8))

    for s in range(size):
        print('\rGenerated %i/%i pairs' % (s, size), end='')

        i, l = generate_training_pair()
        inputs[s] = i
        labels[s] = l

    return inputs.reshape(-1, img_size, img_size, 1), labels



net = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(8)
])

net.summary()
net.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

train_images, train_labels = generate_dataset(5000)
validation_set = generate_dataset(10)

net.fit(train_images, train_labels, epochs=50, validation_data=validation_set)
