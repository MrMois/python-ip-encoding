import numpy as np


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
    pass


def detect_corners(corrected_arr):
    pass


def read_code_pixel(y, x, corrected_arr, corners):
    pass


def read_code_photo(rgb_arr):
    pass
