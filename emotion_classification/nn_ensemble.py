from keras_vgg8 import vgg8
from keras_vgg10 import vgg10
from keras_vgg12 import vgg12
from keras_vgg14 import vgg14
import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    test_y = np.load('../data/fer2013/test_y.npy')
    n, = test_y.shape

    vgg8_pred = np.argmax(vgg8(), axis=1)
    vgg10_pred = np.argmax(vgg10(), axis=1)
    vgg12_pred = np.argmax(vgg12(), axis=1)
    vgg14_pred = np.argmax(vgg14(), axis=1)

    stack = np.vstack([vgg8_pred, vgg10_pred, vgg12_pred, vgg14_pred])

    print(stack.shape)
    counts = [np.bincount(stack[:, i], minlength=7) for i in range(n)]
    predictions = np.argmax(counts, axis=1)
    print(accuracy_score(test_y, predictions))