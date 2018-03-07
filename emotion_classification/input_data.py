import tensorflow as tf
import numpy as np


def read_data():
    file_name = 'fer2013/fer2013.csv'
    dataset = tf.data.TextLineDataset(file_name).skip(1)
    iterator = dataset.make_one_shot_iterator()

