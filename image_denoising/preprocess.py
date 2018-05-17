import numpy as np
from sklearn.feature_extraction import image
from scipy.misc import imread, imresize, imsave
from scipy.ndimage.filters import gaussian_filter
import h5py
from os import listdir

scale = 2
size_input = 33
imagesize = 250
size = 1000 * imagesize


def pre():
    # Localtion of faces in the wild dataset
    filespaths = listdir("lfw/")
    np.random.shuffle(filespaths)
    dataset = h5py.File("dataset_part3.hdf5", "w")
    train_x = dataset.create_dataset("train_x", (size, size_input, size_input, 1))
    train_y = dataset.create_dataset("train_y", (size, size_input, size_input, 1))
    count = 0
    for name in filespaths[0:999]:
        img = imread("lfw/" + name, mode="L")
        img = imresize(img, (imagesize, imagesize), interp="bicubic")
        patches = image.extract_patches_2d(img, (size_input, size_input), max_patches=imagesize)
        for i in range(imagesize):
            ip = patches[i]
            op = gaussian_filter(ip, sigma=0.5)
            op = imresize(op, (size_input // scale, size_input // scale))
            op = imresize(op, (size_input, size_input), interp="bicubic")
            op = np.reshape(op, (33, 33, 1))
            ip = np.reshape(ip, (33, 33, 1))
            train_x[count, :, :, :] = op.astype("float64") / 255.0
            train_y[count, :, :, :] = ip.astype("float64") / 255.0
            count += 1
    imsave("test.png", train_x[0])
    imsave("test2.png", train_x[size - 1])
    dataset.close()


if __name__ == "__main__":
    pre()
