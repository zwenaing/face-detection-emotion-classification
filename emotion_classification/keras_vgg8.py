from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.initializers import Zeros
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score


def vgg8():

    file_path = '../data/fer2013/'
    # train_x = np.load(file_path + "train_X.npy")
    # train_y = np.load(file_path + "train_y.npy")
    # validation_x = np.load(file_path + "validation_X.npy")
    # validation_y = np.load(file_path + "validation_y.npy")
    test_x = np.load(file_path + "test_X.npy")
    test_y = np.load(file_path + "test_y.npy")

    input = Input(shape=[48, 48, 1])
    r1_c1 = Conv2D(filters=32, kernel_size=[3, 3], padding="same", activation="relu", bias_initializer=Zeros())(input)
    r1_p1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(r1_c1)

    r2_c1 = Conv2D(filters=64, kernel_size=[3, 3], padding="same", activation="relu", bias_initializer=Zeros())(r1_p1)
    r2_p1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(r2_c1)

    r3_c1 = Conv2D(filters=128, kernel_size=[3, 3], padding="same", activation="relu", bias_initializer=Zeros())(r2_p1)
    r3_p1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(r3_c1)

    r4_c1 = Conv2D(filters=256, kernel_size=[3, 3], padding="same", activation="relu", bias_initializer=Zeros())(r3_p1)
    r4_c2 = Conv2D(filters=256, kernel_size=[3, 3], padding="same", activation="relu", bias_initializer=Zeros())(r4_c1)
    r4_p1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(r4_c2)

    flatten = Flatten()(r4_p1)
    fcc1 = Dense(256, activation="relu")(flatten)
    dp1 = Dropout(0.5)(fcc1)

    fcc2 = Dense(256, activation="relu")(dp1)
    dp2 = Dropout(0.5)(fcc2)

    logits = Dense(7, activation="softmax")(dp2)

    model = Model(inputs=input, outputs=logits)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], optimizer="adam")

    model.load_weights("../data/vgg8_raw/" + "keras_vgg8.h5")

    datagen = ImageDataGenerator(rotation_range=45, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(train_x)
    train_history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=256), epochs=30, validation_data=(validation_x, validation_y), verbose=1)


    model.save_weights("keras_vgg8.h5")

    print(accuracy_score(test_y, np.argmax(pred, axis=1)))
    print(train_history.history.keys())

    loss_history = train_history.history["loss"]
    acc_history = train_history.history["sparse_categorical_accuracy"]
    val_loss_history = train_history.history["val_loss"]
    val_acc_history = train_history.history["val_sparse_categorical_accuracy"]

    np.savetxt("loss_history.txt", np.array(loss_history))
    np.savetxt("acc_history.txt", np.array(acc_history))
    np.savetxt("val_loss_history.txt", np.array(val_loss_history))
    np.savetxt("val_acc_history.txt", np.array(val_acc_history))


if __name__ == "__main__":
    main()
