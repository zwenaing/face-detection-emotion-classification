import tensorflow as tf
import numpy as np
from emotion_classification import input_data

learning_rate = 0.01
batch_size =256
training_steps = 1000
file_path = '../data/fer2013/'

def cnn_model(features, labels, mode):
    input = tf.cast(tf.reshape(features, [-1, 42, 42, 1]), tf.float32)
    conv1 = tf.layers.conv2d(
        input,
        filters=32,
        kernel_size=[5, 5],
        strides=1,
        activation=tf.nn.relu,
        padding="same")
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=[2, 2],
        strides=2)
    conv2 = tf.layers.conv2d(
        pool1,
        filters=32,
        kernel_size=[4, 4],
        strides=1,
        activation=tf.nn.relu,
        padding="same")
    pool2 = tf.layers.average_pooling2d(
        conv2,
        pool_size=[3, 3],
        strides=2)
    conv3 = tf.layers.conv2d(
        pool2,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        activation=tf.nn.relu,
        padding="same")
    pool3 = tf.layers.average_pooling2d(
        conv3,
        pool_size=[2, 2],
        strides=2)

    flatten_pool3 = tf.reshape(pool3, [-1, 5 * 5 * 64])
    print(flatten_pool3.get_shape())
    fc1 = tf.layers.dense(flatten_pool3, units=3072, activation=tf.nn.relu)
    dropout = tf.layers.dropout(fc1, rate=0.1, training= mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(dropout, units=7)
    print(tf.shape(logits))
    predictions = {
        "probabilities": tf.nn.softmax(logits, axis=1),
        "logits": logits,
        "classes": tf.argmax(logits, axis=1)
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy =  tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = {
            "accuracy": accuracy
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main():
    train_X = np.load(file_path + 'train_X.npy')
    train_y = np.load(file_path + 'train_y.npy')
    train_X = input_data.image_resize(train_X)

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model,
        model_dir='tmp/bkk_start'
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_X,
        y=train_y,
        batch_size=batch_size,
        shuffle=True
    )

    classifier.train(
        input_fn=train_input_fn,
        steps=1000
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()


