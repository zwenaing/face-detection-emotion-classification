import tensorflow as tf
import numpy as np

def dummy_input():
    # data = tf.random_uniform([1000, 33, 33, 3])
    # labels = tf.random_uniform([1000, 21, 21, 1])
    # test_data = tf.random_uniform([10, 33, 33, 3])
    # test_labels = tf.random_uniform([10, 21, 21, 1])
    train_data = np.random.randint(0, 255, (1000, 33, 33, 3), dtype=np.int32)
    train_labels = np.random.randint(0, 255, (1000, 21, 21, 1), dtype=np.int32)
    test_data = np.random.randint(0, 255, (10, 33, 33, 3), dtype=np.int32)
    test_labels = np.random.randint(0, 255, (10, 21, 21, 1), dtype=np.int32)

    return (train_data, train_labels), (test_data, test_labels)

def load_data():
    return dummy_input()
