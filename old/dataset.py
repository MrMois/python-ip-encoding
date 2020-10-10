import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers

# PER DATA SETTINGS
IMAGE_X = 256
IMAGE_Y = 256



def create_code_image(addr_arr):

    data = np.ones((6, 10), dtype=np.uint8) * 255

    border = [(0,2), (0,3), (0,6), (0,7), (5,0), (5,1), (5,4), (5,5), (5,8), (5,9)]

    for b in border:
        data[b] = 0

    addr_np = np.array(addr_arr, dtype=np.uint8)

    for a in range(4):
        for b in range(8):
            data[a+1][b+1] = ((addr_np[a] & (1 << b)) != 0) * 255

    return Image.fromarray(data)


def create_background(color):
    data = np.ones((IMAGE_Y, IMAGE_X), dtype=np.uint8) * color
    return Image.fromarray(data)



def create_training_data():

    bg = create_background(np.floor(np.random.random() * 196))

    addr = np.floor(np.random.randn(4) * 256)
    code = create_code_image(addr)

    scale = 10 + np.random.random() * 10
    degree = np.random.random() * 360

    code = code.resize((int(code.size[0] * scale), int(code.size[1] * scale)), Image.NEAREST)
    mask = Image.new('L', code.size, 255)

    code = code.rotate(degree, expand=True)
    mask = mask.rotate(degree, expand=True)

    off_y = int(np.random.random() * (IMAGE_Y - code.size[0]))
    off_x = int(np.random.random() * (IMAGE_X - code.size[1]))

    bg.paste(code, (off_y, off_x), mask)

    addr_bytes = np.array(addr, dtype=np.uint8)
    addr_vector = np.zeros(32)

    for a in range(4):
        for b in range(8):
            addr_vector[a * 8 + b] = ((addr_bytes[a] & (1 << b)) != 0)

    return bg, addr_vector


def generate_dataset(size):

    inputs = np.zeros((size, IMAGE_Y, IMAGE_X))
    truths = np.zeros((size, 32))

    for i in range(size):

        img, t = create_training_data()

        inputs[i] = np.array(img)
        truths[i] = t

    return inputs, truths



model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_Y, IMAGE_X, 1)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_inputs, train_truths = generate_dataset(512)
train_inputs = train_inputs.reshape(-1, IMAGE_Y, IMAGE_X, 1)
print(train_inputs.shape)

history = model.fit(train_inputs, train_truths, epochs=20)
