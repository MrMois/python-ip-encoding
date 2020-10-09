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


def get_code_image_arr(byte1, byte2, byte3):

    arr = np.ones((5,12,3), dtype=np.uint8) * 255
    # Convention, used for orientation in decoding
    arr[1,1,:] = arr[1,10,:] = np.array([0,0,255])
    arr[3,1,:] = arr[3,10,:] = np.array([255,0,0])

    # Create list of three bit arrays
    bits = [byte_to_bitarr(b) for b in [byte1, byte2, byte3]]

    for c in range(3):
        for b in range(8):
            if(bits[c][b] == 1):
                arr[c+1, b+2, c-1] = 0
                arr[c+1, b+2, c-2] = 0

    return arr


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


def detect_corners(pp_arr, k=4, min_b=0.6):

    (height, width) = pp_arr.shape
    # Order: tl, tr, br, bl
    # Each row: [x, y, b]
    corners = np.zeros((4,3))

    searchspace = (2 * k + 1)**2

    for x in range(k, width-(k+1)):
        for y in range(k, height-(k+1)):
            # Only white pixels can be corner
            if pp_arr[y,x] != 3:
                continue

            b = 0

            for i in range(-k, k+1):
                for j in range(-k, k+1):
                    # Value means black in pp_arr
                    if pp_arr[y+j, x+i] == 4:
                        b += 1

            b /= searchspace

            if b > min_b:
                # Find closest image corner
                cc =    (y < height/2 and x < width/2) * 0 \
                    +   (y < height/2 and x > width/2) * 1 \
                    +   (y > height/2 and x > width/2) * 2 \
                    +   (y > height/2 and x < width/2) * 3

                # Check if new best corner found
                if b > corners[cc][2]:
                    corners[cc] = np.array([x, y, b])

    # Discard b values of final corners
    return corners[:,:-1].astype(np.int)


def extract_code_pixel(x, y, pp_arr, corners):
    assert(x >= 0 and x < 12)
    assert(y >= 0 and y < 5)

    (height, width) = pp_arr.shape

    tl = corners[0]
    tr = corners[1]
    br = corners[2]
    bl = corners[3]

    sl = tl + (1 + 2*y) / 10 * (bl - tl)
    sr = tr + (1 + 2*y) / 10 * (br - tr)
    px = sl + (1 + 2*x) / 24 * (sr - sl)
    px = np.floor(px + 0.5).astype(np.int)

    return pp_arr[px[1], px[0]]


def extract_bytes_from_photo(rgb_arr):

    pp = preprocessing(rgb_arr)
    corners = detect_corners(pp)

    code = np.zeros((5,12))
    # Read code pixels (note: they are in 0-4 format)
    for y in range(5):
        for x in range(12):
            code[y,x] = extract_code_pixel(x, y, pp, corners)

    if not (code[1, 1] == 2
        and code[1,10] == 2
        and code[3, 1] == 0
        and code[3,10] == 0):
        return 'INVALID_CODE_CHECKPOINTS'

    bytes = []

    for byte in range(3):

        bits = []

        for bit in range(8):
            if code[byte+1, bit+2] == byte:
                bits.append(1)
            elif code[byte+1, bit+2] == 3:
                bits.append(0)
            else:
                return 'INVALID_CODE_BYTES'

        bytes.append(bitarr_to_byte(np.array(bits)))

    return bytes


code_arr = get_code_image_arr(15,127,255)
img = Image.fromarray(code_arr)
img = img.resize((img.size[0] * 10, img.size[1] * 10), Image.NEAREST)

big = Image.new('RGB', (200, 200), 0)
big.paste(img, (30,90))
big.show()

big_arr = np.array(big)

bytes = extract_bytes_from_photo(big_arr)
print(bytes)
