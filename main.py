from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math


def log_2(x):
    if x == 0:
        return False
    return math.log10(x) / math.log10(2)


def is_power_of_2(n):
    return math.ceil(log_2(n)) == math.floor(log_2(n))


def dither_matrix(size):
    window_m1 = 1/4 * np.matrix([[0, 2], [3, 1]])

    if not is_power_of_2(size) and size < 2:
        print('size should be power of 2')
        exit()
    else:
        n = math.log2(size)
        new_n = 1
        new_window = window_m1
        while new_n != n:
            new_n += 1
            two_n_two = (2 * new_n) ** 2

            new_window = 1/two_n_two * np.block([[two_n_two * new_window, two_n_two * new_window + 2],
                                                 [two_n_two * new_window + 3, two_n_two * new_window + 1]])
        return new_window


img = mpimg.imread('solar-opposites.jpg')

R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
image_gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

# plt.imshow(image_gray, cmap='gray')
# plt.show()

window_size = int(input('window size: '))
dither_window = dither_matrix(window_size)

new_image = np.zeros((window_size * len(image_gray), window_size * len(image_gray[0])))


for i in range(len(image_gray)):
    for j in range(len(image_gray[0])):
        sub_value = image_gray[i][j] / 255
        for k in range(window_size):
            for p in range(window_size):
                if sub_value > dither_window[k, p]:
                    new_image[i * window_size + k][j * window_size + p] = 1

plt.imshow(new_image, cmap='gray')
plt.show()

