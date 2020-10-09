"""
Decoding photo algorithm
1. Preprocessing:
    Change every pixel to nearest matching color (Red,Green,Blue,Black,White)
2. Find corners:
    Corner is point where surrounding pixels are >50% black (>10% white?)
3. Crop and perspective transform
4. Decode code image
"""

import numpy as np
from PIL import Image


def byte_to_bitarr(byte):
    assert(byte <= 255 and byte >= 0)

    arr = np.zeros(8)
    byte = np.uint8(byte)

    for b in range(8):
        if (byte & (1 << 7-b)) != 0:
            arr[b] = 1

    return arr

def code_img_from_rgb(r, g, b):

    # Create pixel array
    arr = np.ones((5, 10, 3), dtype=np.uint8) * 255
    # Convert values to bytes
    bits = [byte_to_bitarr(v) for v in [r,g,b]]

    for c in range(3):
        for b in range(8):
            if(bits[c][b] == 1):
                arr[c+1, b+1, c-1] = 0
                arr[c+1, b+1, c-2] = 0

    return Image.fromarray(arr)


def color_correction(rgb_arr):

    (height, width, depth) = rgb_arr.shape

    c = np.array([[255,0,0], [0,255,0], [0,0,255], [0,0,0], [255,255,255]])

    for y in range(height):
        for x in range(width):
            min_i = 0
            min = np.inf
            for i in range(5):
                dist = np.linalg.norm(rgb_arr[y,x,:] - c[i])
                if dist < min:
                    min_i = i
                    min = dist
            rgb_arr[y,x,:] = c[min_i]

    return rgb_arr


def discard_color(rgb_arr, thresh=128):

    (height, width, depth) = rgb_arr.shape

    bw_arr = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if rgb_arr[y,x,:].sum() >= thresh:
                bw_arr[y,x] = 255

    return bw_arr


# Corner order: TL, TR, BR, BL
def detect_corners(bw_arr):

    (height, width) = bw_arr.shape

    # Kernel
    k = 4
    # Searchspace
    s = (2 * k + 1)**2

    MIN_B = 0.5

    corners = np.zeros((4,3))

    for y in range(k, height-k):
        for x in range(k, width-k):
            if not bw_arr[y,x] == 255:
                continue
            # Counter variable
            b = 0
            # Loop kernel
            for y_k in range(-k, k):
                for x_k in range(-k, k):
                    if bw_arr[y+y_k,x+x_k] == 0:
                        b += 1

            r_b = b / s

            if r_b > MIN_B:
                # Found new corner candidate

                # TODO: Search for closest corner in candidates, not the other way around!

                # Find closest image corner
                i =     (y < height/2 and x < width/2) * 0 \
                    +   (y < height/2 and x > width/2) * 1 \
                    +   (y > height/2 and x > width/2) * 2 \
                    +   (y > height/2 and x < width/2) * 3 \

                # Check if more black than previous candidate
                if r_b > corners[i][2]:
                    corners[i] = np.array([y,x,r_b])

    return corners[:,:2]


def extract_code_from_image(cor_rgb_arr, corners):

    (height, width, depth) = cor_rgb_arr.shape
    assert(depth == 3)

    code = np.zeros((3,8))
    # TODO: Add probability for each code bit (by searching in kernel area)

    tl = corners[0]
    tr = corners[1]
    br = corners[2]
    bl = corners[3]

    colors = np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,255]])

    # TODO: Remove!!
    debug = Image.fromarray(cor_rgb_arr)

    # Iterate colors (rows)
    for c in range(3):

        sl = tl + (3 + 2*c) / 10 * (bl - tl)
        sr = tr + (3 + 2*c) / 10 * (br - tr)

        # Iterate bits (columns)
        for b in range(8):

            p = sl + (3 + 2*b) / 20 * (sr - sl)
            p = np.floor(p).astype(np.int)

            print('sl, sr, p', sl, sr, p)

            point = Image.new('RGB', (10,10), 128)
            debug.paste(point, (p[1], p[0]))


            pixel = cor_rgb_arr[p[0], p[1],:]

            if (pixel == colors[c]).all():
                code[c,b] = 1
            elif not (pixel == colors[3]).all():
                code[c,b] = -1
                
    debug.show()

    return code



fg_rand = np.zeros((4,4,3))

for y in range(4):
    for x in range(4):
        v = np.random.rand(3)
        fg_rand[y,x,:] = v / np.linalg.norm(v) * 255

fg_rand = fg_rand.astype(np.uint8)


bg = Image.new('RGB', (300, 200), 0)

fg = Image.new('RGB', (100, 100), (255, 255, 255))
fg = Image.fromarray(fg_rand)
fg = fg.resize((fg.size[0]  * 25, fg.size[1]  * 25), Image.NEAREST)
fg = fg.rotate(10, expand=True)

bg.paste(fg, (130, 70))
bg.show()

bg_arr = np.array(bg)

cor = color_correction(bg_arr)
cor_img = Image.fromarray(cor)
cor_img.show()

bw = discard_color(bg_arr, thresh=128)
bw_img = Image.fromarray(bw)
bw_img.show()

corners = detect_corners(bw)

print('Corners', corners)

dot_size = 10
dot = Image.new('RGB', (dot_size, dot_size), (255,0,0))

for i in range(4):
    if corners[i][0] > 0 and corners[i][1] > 0:
        bw_img.paste(dot, (int(corners[i][1] - dot_size / 2), int(corners[i][0] - dot_size / 2)))

bw_img.show()

code = extract_code_from_image(cor, corners)
print('Code', code)
