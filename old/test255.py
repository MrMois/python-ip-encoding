import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers


IMG_SIZE = 164


def create_code_img(number):

    arr = np.full((4,4,4), (0,0,0,255),dtype=np.uint8)

    byte = np.uint8(number)

    for i in range(8):
        if ((byte & (1 << i)) != 0):
            arr[int(i / 4), i % 4] = np.ones(4) * 255
        else:
            arr[int(i / 4), i % 4] = np.zeros(4)

    return Image.fromarray(arr)


def create_bit_vector(number):

    arr = np.zeros(8)

    byte = np.uint8(number)

    for i in range(8):
        arr[i] = ((byte & (1 << i)) != 0)

    return arr


def create_training_img(number, color):

    code = create_code_img(number)

    scale = 6 + np.random.random() * 12
    code = code.resize((int(code.size[0] * scale), int(code.size[1] * scale)), Image.NEAREST)

    degree = np.random.random() * 360
    code = code.rotate(degree, expand=True)

    off_y = int(np.random.random() * (IMG_SIZE - code.size[0]))
    off_x = int(np.random.random() * (IMG_SIZE - code.size[1]))

    res = Image.new('RGBA', (IMG_SIZE, IMG_SIZE), color)
    res.paste(code, (off_y, off_x), code)

    return res


def create_dataset(size, grayscale=True):

    inputs = np.zeros((size, IMG_SIZE, IMG_SIZE))
    labels = np.zeros((size, 8))

    for i in range(size):

        n = np.floor(np.random.random() * 8)

        col = np.random.randint(3)
        col = ((col==0) * 255, (col==1) * 255, (col==2) * 255, 255)

        img = create_training_img(n, col)
        img = img.convert('L')
        label = create_bit_vector(n)

        inputs[i] = np.array(img)
        labels[i] = label

    return inputs, labels


model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='relu'),
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_inputs, train_truths = create_dataset(512)
train_inputs = train_inputs.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

history = model.fit(train_inputs, train_truths, epochs=20)
