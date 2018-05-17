import numpy as np
from scipy.misc import imresize, imsave, imread
from train import processImg

scale = 2


def readImg(filename):
    img = imread(filename, mode="L")
    shape = np.shape(img)
    bicubic = imresize(img, (shape[0] * scale, shape[1] * scale), interp="bicubic")
    bicubic = np.reshape(bicubic, (500, 500, 1))
    temp = np.zeros((1, shape[0] * scale, shape[1] * scale, 1))
    temp[0, :, :, :] = bicubic.astype("float64") / 255.0
    result = processImg(temp, shape[0] * scale, shape[1] * scale, k=1, weight="srcnn_128_64_1_9_1_5.h5")
    result = result[0, :, :, :].astype("float64") * 255
    result = np.reshape(result, (500, 500))
    result = np.clip(result, 0, 255).astype("uint8")
    imsave("128_9_1_5_srcnn.bmp", result)


if __name__ == "__main__":
    readImg("original.jpg")
    img = imread("C:\\Users\\Cheng\\Desktop\\SRCNN_Part2\\original.jpg", mode="L")
    img = imresize(img, (500,500), interp="bicubic")
    imsave("bicubic.bmp",img)
