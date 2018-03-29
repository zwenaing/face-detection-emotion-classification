import tensorflow as tf
from os.path import exists
import numpy as np
from input_data import image_resize

learning_rate = 0.1
momentum = 0.9
batch_size = 256
display_steps = 10
num_steps = 10000

log_dir = "tmp/bkvgg8"
file_path = "../data/fer2013/"

image_size = 42
num_channels = 1
num_classes = 7

X = tf.placeholder(tf.float32, [None, image_size * image_size * num_channels])
y = tf.placeholder(tf.int32, [None])

with tf.name_scope("input_layer"):
    input_layer = tf.reshape(X, [-1, image_size, image_size, num_channels])

with tf.name_scope("rect_1"):
    with tf.name_scope("conv3_32_1"):
        rect1_conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu
        )
    with tf.name_scope("max_pool"):
        rect1_max_pool = tf.layers.max_pooling2d(
            inputs=rect1_conv1,
            pool_size=[2, 2],
            strides=2,
            padding="valid"
        )

with tf.name_scope("rect_2"):
    with tf.name_scope("conv3_64_1"):
        rect2_conv1 = tf.layers.conv2d(
            inputs=rect1_max_pool,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu
        )
    with tf.name_scope("max_pool"):
        rect2_max_pool = tf.layers.max_pooling2d(
            inputs=rect2_conv1,
            pool_size=[2, 2],
            strides=2,
            padding="valid"
        )

with tf.name_scope("rect_3"):
    with tf.name_scope("conv3_128_1"):
        rect3_conv1 = tf.layers.conv2d(
            inputs=rect2_max_pool,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu
        )
    with tf.name_scope("max_pool"):
        rect3_max_pool = tf.layers.max_pooling2d(
            inputs=rect3_conv1,
            pool_size=[2, 2],
            strides=2,
            padding="valid"
        )

with tf.name_scope("rect_4"):
    with tf.name_scope("conv3_256_1"):
        rect4_conv1 = tf.layers.conv2d(
            inputs=rect3_max_pool,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu
        )
    with tf.name_scope("conv3_256_2"):
        rect4_conv2 = tf.layers.conv2d(
            inputs=rect4_conv1,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu
        )
    with tf.name_scope("flatten_units"):
        flatten_units = tf.reshape(rect4_conv2, [-1, 5 * 5 * 256])

with tf.name_scope("fccs"):
    with tf.name_scope("fcc_1"):
        fcc1 = tf.layers.dense(
            inputs=flatten_units,
            units=256,
            activation=tf.nn.relu
        )
    with tf.name_scope("fcc_2"):
        fcc2 = tf.layers.dense(
            inputs=fcc1,
            units=256,
            activation=tf.nn.relu
        )

with tf.name_scope("logits"):
    logits = tf.layers.dense(
        inputs=fcc2,
        units=num_classes,
        activation=tf.nn.relu
    )

with tf.name_scope("softmax"):
    probabilities = tf.nn.softmax(logits, axis=1)

with tf.name_scope("predictions"):
    predictions = tf.argmax(probabilities, axis=1)

with tf.name_scope("softmax_loss"):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        #(onehot_labels=tf.one_hot(y, num_classes), logits=logits)

with tf.name_scope("accuracy"):
    accuracy, _ = tf.metrics.accuracy(labels=y, predictions=predictions)

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("optimization"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run([init, local_init])

    saver.restore(sess, log_dir)

    train_X = np.load(file_path + 'resized_train_X.npy')
    train_y = np.load(file_path + 'train_y.npy')
    features_placeholder = tf.placeholder(train_X.dtype, train_X.shape)
    labels_placeholder = tf.placeholder(train_y.dtype, train_y.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    train_dataset = train_dataset.shuffle(1000).repeat().batch(batch_size)
    iterator = train_dataset.make_initializable_iterator()
    next_items = iterator.get_next()

    sess.run(iterator.initializer, feed_dict={features_placeholder: train_X, labels_placeholder: train_y})

    file_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())

    for i in range(num_steps + 1):

        batch_x, batch_y = sess.run(next_items)
        _, summary = sess.run([train_op, merged_summary], feed_dict={X: batch_x, y: batch_y})

        if i % display_steps == 0:
            file_writer.add_summary(summary)
            loss_, accuracy_ = sess.run([loss, accuracy], feed_dict={X: batch_x, y: batch_y})
            print("Iteration Number: ", str(i), " Loss: ", str(loss_), " Accuracy: ", str(accuracy_))
            saver.save(sess, log_dir)



