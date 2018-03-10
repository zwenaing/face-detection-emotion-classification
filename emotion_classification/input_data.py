import tensorflow as tf
import numpy as np
from skimage.transform import resize

file_path = '../data/fer2013/'

def parse_data():
    data = np.loadtxt(file_path + '/fer2013.csv', skiprows=1, delimiter=',', dtype=np.str)
    # Extract different data sets
    train_data = data[np.argwhere(data[:, 2] == 'Training').ravel()]
    validation_data = data[np.argwhere(data[:, 2] == 'PrivateTest').ravel()]
    test_data = data[np.argwhere(data[:, 2] == "PublicTest").ravel()]
    train_X = train_data[:, 1]
    validation_X = validation_data[:, 1]
    test_X = test_data[:, 1]
    train_n, = train_X.shape
    validation_n, = validation_X.shape
    test_n, = test_X.shape
    train_X = np.array([[int(j) for j in train_X[i].split(' ')] for i in range(train_n)])
    validation_X = np.array([[int(j) for j in validation_X[i].split(' ')] for i in range(validation_n)])
    test_X = np.array([[int(j) for j in test_X[i].split(' ')] for i in range(test_n)])
    train_y = train_data[:, 0].astype(np.int32)
    validation_y = validation_data[:, 0].astype(np.int32)
    test_y = test_data[:, 0].astype(np.int32)
    np.save(file_path + "train_X.npy", train_X)
    np.save(file_path + "train_y.npy", train_y)
    np.save(file_path + "validation_X.npy", validation_X)
    np.save(file_path + "validation_y.npy", validation_y)
    np.save(file_path + "test_X.npy", test_X)
    np.save(file_path + "test_y.npy", test_y)


def image_resize(images):
    n, d = images.shape
    images = images.reshape([-1, 48, 48, 1])
    resized_images = [resize(images[i], [42, 42]) for i in range(n)]
    resized_images = np.reshape(resized_images, [n, -1])
    return resized_images

if __name__ == '__main__':
    train_X = np.load('../data/fer2013/' + 'test_X.npy')
    images = image_resize(train_X[:10, :])
    print(images.shape)



