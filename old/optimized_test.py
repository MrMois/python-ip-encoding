import numpy as np
import cv2
from PIL import Image
from optimized import *

cam = cv2.VideoCapture(1)

def get_new_random_bytes():
    bytes = np.random.randint(0, 256, 3)
    print('New random bytes: ', bytes)
    code_arr = get_code_image_arr(bytes[0], bytes[1], bytes[2])
    scale = 40
    img = Image.fromarray(code_arr)
    img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.NEAREST)
    img.show()

get_new_random_bytes()

while True:

    s, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    (height, width, _) = img.shape

    preview = img[275:365,90:390]

    if is_code_photo_candidate(preview):
        print('Starting scan!')
        toscan = img[220:420]
        bytes = extract_bytes_from_photo(toscan)
        print('Extracted: ', bytes)
        get_new_random_bytes()

    img[275:305,90:120,2] = np.ones((30,30)) * 255
    img[275:305,360:390,2] = np.ones((30,30)) * 255
    img[335:365,90:120,0] = np.ones((30,30)) * 255
    img[335:365,360:390,0] = np.ones((30,30)) * 255


    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('preview', img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
