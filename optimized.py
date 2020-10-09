import numpy as np
from PIL import Image


def byte_to_bitarr(byte):
    assert(byte <= 255 and byte >= 0)

    byte = np.uint8(byte)
    bits = [(byte & (1 << 7-b)) != 0 for b in range(8)]

    return np.array(bits, dtype=np.uint8)


def bitarr_to_byte(bitarr):
    assert(bitarr.shape == (8,))

    twos = [(1 << 7-b) for b, v in enumerate(bitarr) if v == 1]

    return np.sum(twos, dtype=np.uint8)


def get_code_image(byte1, byte2, byte3):
    pass


def preprocessing(rgb_arr):

    (height, width, depth) = rgb_arr.shape
    assert(depth == 3)

    pp_arr = np.zeros((height, width), dtype=np.uint8)
    colors = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0]]) * 255

    for x in range(width):
        for y in range(height):

            min = np.inf
            idx = 0

            dists = [np.linalg.norm(rgb_arr[y,x,:] - c) for c in colors]
            pp_arr[y,x] = np.argmin(dists)

    return pp_arr


def detect_corners(pp_arr, k=4, min_b=0.5):

    (height, width) = pp_arr.shape
    # Order: tl, tr, br, bl
    # Each row: [x, y, b]
    corners = np.zeros((4,3))

    for x in range(k, width-k):
        for y in range(k, height-k):

            b = 0

            for i in range(-k, k):
                for j in range(-k, k):
                    # Value 3 means black in pp_arr
                    if pp_arr[y+j, x+i] == 3:
                        b += 1

            b /= (2 * k + 1)**2

            if b > min_b:
                # Find closest image corner
                cc =    (y < height/2 and x < width/2) * 0 \
                    +   (y < height/2 and x > width/2) * 1 \
                    +   (y > height/2 and x > width/2) * 2 \
                    +   (y > height/2 and x < width/2) * 3 \
                # Check if new best corner found
                if b > corners[cc][2]:
                    corners[cc] = np.array([x, y, b])

    # Discard b values of final corners
    return corners[:,:-1]



def read_code_pixel(y, x, corrected_arr, corners):
    pass


def read_code_photo(rgb_arr):
    pass


r = np.random.rand(4,4,3)

for y in range(4):
    for x in range(4):
        r[y,x,:] = r[y,x,:] / np.linalg.norm(r[y,x,:]) * 255

r = r.astype(np.uint8)

print(r)

i1 = Image.fromarray(r)
i1.show()

p = preprocessing(r)

print(p)
