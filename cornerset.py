import numpy as np
import cv2
import timeit


"""
code = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.uint8) * 255

code = cv2.resize(code, (500, 300), interpolation=cv2.INTER_NEAREST)

background = np.zeros((720, 1080))
background[210:510,290:790] = code

cv2.imwrite('example_code.png', background)
"""


def generate_training_pair(size, min_corner_off, max_edge_off):

    img = np.zeros((size, size))
    label = np.zeros(2*size)

    point = np.random.randint(min_corner_off, size-min_corner_off, 2)

    label[point[0]] = 1
    label[point[1] + size] = 1

    corners = np.array([
        [0, 0],
        [point[0]+np.random.randint(-max_edge_off, max_edge_off), 0],
        point,
        [0, point[1]+np.random.randint(-max_edge_off, max_edge_off)],
    ], dtype=np.int)

    cv2.fillPoly(img, pts=[corners], color=(255, 255, 255))

    return img, point


def img_rgb_to_bw(img, thresh):

    (height, width, depth) = img.shape
    assert(depth == 3)

    bw = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):

            if img[y,x].sum() > thresh:
                bw[y,x] = 255

    return bw


if __name__ == '__main__':

    cam = cv2.VideoCapture(1)
    # Cheat!
    (height, width, depth) = (480, 640, 3)

    while True:

        _, img = cam.read()
        img = img[:,80:560]
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_NEAREST)
        img = img_rgb_to_bw(img, thresh=511)

        # img = cv2.cvtColor(img, cv2.   )
        # dst = cv2.cornerHarris(img, 2, 3, 0.04)

        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('live', img)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
