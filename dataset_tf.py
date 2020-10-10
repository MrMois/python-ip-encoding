import cv2
import numpy as np
import imutils


def byte_to_bitarr(byte):
    assert(byte <= 255 and byte >= 0)

    byte = np.uint8(byte)
    bits = [(byte & (1 << 7-b)) != 0 for b in range(8)]

    return np.array(bits, dtype=np.uint8)


def bytes_to_code(bytes):
    assert(len(bytes) == 4)

    bitarrs = [byte_to_bitarr(b) for b in bytes]
    # Create empty code array
    code = np.ones((6, 10), dtype=np.uint8) * 255
    code[0, 0] = 0
    # Encode bytes
    for row, byte in enumerate(bitarrs):
        for col, bit in enumerate(byte):
            if bit == 1:
                code[row+1, col+1] = 0

    return code


def img_rotate(img, degree):
    # Wrapper for imutils
    return imutils.rotate_bound(img, degree)


def img_paste(img_src, img_tgt, tl=(0, 0)):
    # TODO: Add some dim checks
    (h, w) = img_src.shape
    # Paste into new copied array
    res = img_tgt.copy()
    res[tl[1]:tl[1]+h, tl[0]:tl[0]+w] = img_src

    return res


def img_scale(img, scale, interpol=cv2.INTER_NEAREST):
    return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0] * scale)), interpolation=interpol)


def generate_training_pair(scale=(15,18), rotate=(-10, 10), transform=(0,8), input=(200, 200)):

    bytes = np.random.randint(0, 256, 4)
    label = np.array([byte_to_bitarr(b) for b in bytes])
    label = label.reshape((4*8))

    code = bytes_to_code(bytes)

    scale = np.random.randint(scale[0], scale[1], 1)
    transform = np.random.randint(transform[0], transform[1], 8)
    rotate = np.random.randint(rotate[0], rotate[1], 1)

    code = img_scale(code, scale)
    (h, w) = code.shape

    src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ], dtype=np.float32)

    tgt = np.array([
        [transform[0], transform[1]],
        [w-transform[2], transform[3]],
        [w-transform[4], h-transform[5]],
        [transform[6], h-transform[7]],
    ], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src, tgt)
    code = cv2.warpPerspective(code, mat, (w, h))

    code = img_rotate(code, rotate)

    offset = (
        # x
        np.random.randint(0, input[1]-code.shape[1]),
        # y
        np.random.randint(0, input[0]-code.shape[0]),
    )

    image = np.zeros(input)
    image = img_paste(code, image, offset)
    image = img_scale(image, 0.2)
    image = image / 255.0

    return image, label


def generate_dataset(size):
    # TODO: pass params to generate_training_pair

    inputs = np.zeros((size, 40, 40))
    labels = np.zeros((size, 4*8))

    for s in range(size):
        print('\rGenerated %i/%i pairs' % (s, size), end='')

        i, l = generate_training_pair()
        inputs[s] = i
        labels[s] = l

    return inputs.reshape(-1, 40, 40, 1), labels


if __name__ == '__main__':

    img, label = generate_training_pair()
    print(img.shape)
    print(label)

    cv2.imshow('input', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
