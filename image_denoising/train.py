from keras.layers import Input, Conv2D
from keras.models import Model
from keras import backend as K
from keras.utils import HDF5Matrix
import matplotlib.pyplot as plt


def PSNRLoss(y_true, y_pred):
    return 10. * K.log(1.0 / K.mean(K.square(y_pred - y_true))) / K.log(10.0)


def main(k=5):
    train_x = HDF5Matrix("dataset_part3.hdf5", "train_x")
    train_y = HDF5Matrix("dataset_part3.hdf5", "train_y")
    x = Input(shape=[33, 33, 1])
    c1 = Conv2D(filters=128, kernel_size=(9, 9), activation="relu", init="he_normal", padding="same")(x)
    c2 = Conv2D(filters=64, kernel_size=(k, k), activation="relu", init="he_normal", padding="same")(c1)
    c3 = Conv2D(filters=1, kernel_size=(5, 5), activation=None, init="he_normal", padding="same")(c2)
    model = Model(inputs=x, outputs=c3)
    model.compile(loss='mse', metrics=[PSNRLoss], optimizer="adam")
    history = model.fit(train_x, train_y, batch_size=256, nb_epoch=10, verbose=1, validation_split=0.1)
    model.save_weights("srcnn_128_64_1_9_" + str(k) + "_5.h5")
    plt.plot(history.history['PSNRLoss'])
    plt.plot(history.history['val_PSNRLoss'])
    plt.title('PSNR 128-64-1 9-' + str(k) + '-5')
    plt.ylabel('PSNR/dB')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()


def processImg(img, h, w, k=5, weight="srcnn_9_1_5.h5"):
    x = Input(shape=[h, w, 1])
    c1 = Conv2D(filters=128, kernel_size=(9, 9), activation="relu", init="he_normal", padding="same")(x)
    c2 = Conv2D(filters=64, kernel_size=(k, k), activation="relu", init="he_normal", padding="same")(c1)
    c3 = Conv2D(filters=1, kernel_size=(5, 5), activation=None, init="he_normal", padding="same")(c2)
    model = Model(inputs=x, outputs=c3)
    model.load_weights(weight)
    pred = model.predict(img)

    return pred


if __name__ == "__main__":
    main(k=1)
