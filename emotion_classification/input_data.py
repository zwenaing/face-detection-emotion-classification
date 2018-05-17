import tensorflow as tf
import numpy as np
from skimage.transform import resize
from sklearn.metrics import accuracy_score
import math

file_path = '../data/fer2013/'

def read_txt():
    with open('human_labels.txt') as f:
        str = f.read()
    array = str.split('\n')
    int_array = [int(array[i]) - 1 for i in range(len(array))]
    return int_array

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

    np.save(file_path + "train_X.npy", np.reshape(train_X, [-1, 48, 48, 1]))
    np.save(file_path + "train_y.npy", train_y)
    np.save(file_path + "validation_X.npy", np.reshape(validation_X, [-1, 48, 48, 1]))
    np.save(file_path + "validation_y.npy", validation_y)
    np.save(file_path + "test_X.npy", np.reshape(test_X, [-1, 48, 48, 1]))
    np.save(file_path + "test_y.npy", test_y)


def image_resize(images):
    n, d = images.shape
    images = images.reshape([-1, 48, 48, 1])
    resized_images = [resize(images[i], [42, 42]) for i in range(n)]
    resized_images = np.reshape(resized_images, [n, -1])
    return resized_images


def per_image_normalization(data, std):
    mean = np.mean(data, axis=1).reshape([-1, 1])
    std_dev = np.std(data, axis=1).reshape([-1, 1]) + 0.00001
    res = std * ((data - mean) / std_dev)
    return res


def per_pixel_normalization(data):
    mean = np.mean(data, axis=0).reshape([1, -1])
    std = np.std(data, axis=0).reshape([1, -1])
    res = (data - mean) / std
    return res


def augment_images(image, label):
    resized_image = tf.reshape(image, [48, 48, 1])
    fliped_image = tf.image.random_flip_left_right(resized_image)
    rand_deg = np.random.randint(0, 90) - 45
    #rotated_image = tf.contrib.image.rotate(fliped_image, rand_deg * math.pi / 180)

    crop_size = 42
    rescaled_size = np.random.randint(43, 54)
    offset = np.random.randint(0, rescaled_size - crop_size)
    rescaled_image = tf.image.resize_images(fliped_image, [rescaled_size, rescaled_size])
    cropped_image = tf.image.crop_to_bounding_box(rescaled_image, offset, offset, crop_size, crop_size)
    return tf.reshape(cropped_image, [1, -1]), label


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat()
    # dataset = dataset.map(augment_images)
    #dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=augment_images, batch_size=batch_size))
    # dataset = dataset.prefetch(prefetch_batch_size)
    dataset = dataset.batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        input = features
    else:
        input = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(input)

    assert batch_size is not None, "batch size must not be None"

    return dataset.batch(batch_size)


if __name__ == '__main__':
    # parse_data()
    #int_array = read_txt()
    #test_y = np.load("../data/fer2013/test_y.npy")
    #print(accuracy_score(test_y[:500].tolist(), int_array))

    train_X = np.load('../data/fer2013/' + 'denoised_train_X.npy')
    validation_X = np.load('../data/fer2013/' + 'denoised_validation_X.npy')
    test_X = np.load('../data/fer2013/' + 'denoised_test_X.npy')

    train_X = np.reshape(train_X, [-1, 48 * 48])
    validation_X = np.reshape(validation_X, [-1, 48 * 48])
    test_X = np.reshape(test_X, [-1, 48 * 48])

    train_res = per_image_normalization(train_X, 3.125)
    train_res = per_pixel_normalization(train_res)

    validation_res = per_image_normalization(validation_X, 3.125)
    validation_res = per_pixel_normalization(validation_res)

    test_res = per_image_normalization(test_X, 3.125)
    test_res = per_pixel_normalization(test_res)

    np.save("../data/fer2013/denoised_normalized_train_X.npy", np.reshape(train_res, [-1, 48, 48, 1]))
    np.save("../data/fer2013/denoised_normalized_validation_X.npy", np.reshape(validation_res, [-1, 48, 48, 1]))
    np.save("../data/fer2013/denoised_normalized_test.npy", np.reshape(test_res, [-1, 48, 48, 1]))

    # train_x = np.load("../data/fer2013/normalized_train_X.npy")
    # test_x = np.load("../data/fer2013/normalized_validation_X.npy")
    # print(np.mean(train_x, axis=0))
    # print(np.std(train_x, axis=0))
    # print(train_x.shape)
    # print(test_x.shape)
    # train_x = np.load("../data/fer2013/normalized_train_X.npy")
    # train_y = np.load("../data/fer2013/train_y.npy")
    # one_hot_train_y = np.eye(7)[train_y]
    #one_hot_train_y[:, train_y] = 1

    # print(one_hot_train_y.shape)
    # delete_indices = np.where(train_y == 0)[0][3000:]
    # train_x = np.delete(train_x, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    # delete_indices = np.where(train_y == 2)[0][3000:]
    # train_x = np.delete(train_x, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    # delete_indices = np.where(train_y == 3)[0][3000:]
    # train_x = np.delete(train_x, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    # delete_indices = np.where(train_y == 4)[0][3000:]
    # train_x = np.delete(train_x, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    # delete_indices = np.where(train_y == 5)[0][3000:]
    # train_x = np.delete(train_x, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    # delete_indices = np.where(train_y == 6)[0][3000:]
    # train_x = np.delete(train_x, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    #
    # delete_indices = np.where(train_y == 3)[0][-3215:]
    # train_X = np.delete(train_X, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    # delete_indices = np.where(train_y == 4)[0][-830:]
    # train_X = np.delete(train_X, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)
    # delete_indices = np.where(train_y == 6)[0][-965:]
    # train_X = np.delete(train_X, delete_indices, axis=0)
    # train_y = np.delete(train_y, delete_indices, axis=0)

    # print(train_x.shape)
    # print(np.bincount(train_y))
    #
    # train_y = np.load("../data/fer2013/validation_y.npy")
    #
    # print(train_y.shape)
    # print(np.bincount(train_y))
    # train_y = np.load("../data/fer2013/test_y.npy")
    # print(train_y.shape)
    # print(np.bincount(train_y))

