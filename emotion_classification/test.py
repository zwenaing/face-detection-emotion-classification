from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from input_data import train_input_fn


learning_rate = 0.01
momentum = 0.9
batch_size = 32
training_steps = 3000
save_steps = 100
file_path = '../data/fer2013/'
log_dir = 'tmp/bkk_test/'


def model(features, labels, mode, params):

    input = tf.cast(tf.reshape(features, [-1, 42, 42, 1]), tf.float32)

    rect1_conv1 = tf.layers.conv2d(
        inputs=input,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer()
    )

    mean, variance = tf.nn.moments(rect1_conv1, [0, 1, 2])
    rect1_normalized_conv1 = tf.nn.batch_normalization(rect1_conv1, mean, variance, None, None, 0.0001)

    rect1_max_pool = tf.layers.max_pooling2d(
        inputs=rect1_normalized_conv1,
        pool_size=[3, 3],
        strides=2,
        padding="valid"
    )

    rect2_conv1 = tf.layers.conv2d(
        inputs=rect1_max_pool,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer()
    )

    mean, variance = tf.nn.moments(rect2_conv1, [0, 1, 2])
    rect2_normalized_conv1 = tf.nn.batch_normalization(rect2_conv1, mean, variance, None, None, 0.0001)

    rect2_max_pool = tf.layers.max_pooling2d(
        inputs=rect2_normalized_conv1,
        pool_size=[3, 3],
        strides=2,
        padding="valid"
    )

    rect3_conv1 = tf.layers.conv2d(
        inputs=rect2_max_pool,
        filters=128,
        kernel_size=[4, 4],
        strides=1,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer()
    )

    mean, variance = tf.nn.moments(rect3_conv1, [0, 1, 2])
    rect3_normalized_conv1 = tf.nn.batch_normalization(rect3_conv1, mean, variance, None, None, 0.0001)
    #
    # rect3_max_pool = tf.layers.max_pooling2d(
    #     inputs=rect3_normalized_conv1,
    #     pool_size=[2, 2],
    #     strides=2,
    #     padding="valid"
    # )
    #
    # rect4_conv1 = tf.layers.conv2d(
    #     inputs=rect3_max_pool,
    #     filters=64,
    #     kernel_size=[3, 3],
    #     strides=1,
    #     padding="valid",
    #     activation=tf.nn.relu
    # )
    #
    # mean, variance = tf.nn.moments(rect4_conv1, [0, 1, 2])
    # rect4_normalized_conv1 = tf.nn.batch_normalization(rect4_conv1, mean, variance, None, None, 0.0001)

    flatten_units = tf.reshape(rect3_normalized_conv1, [batch_size, 3 * 3 * 128])

    fcc1 = tf.layers.dense(
        inputs=flatten_units,
        units=1024,
        activation=tf.nn.relu
    )
    fcc1 = tf.layers.dropout(fcc1, 0.5, training= mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(
        inputs=fcc1,
        units=7,
        activation=tf.nn.softmax
    )

    predictions = {
        "probabilities": tf.nn.softmax(logits, axis=1),
        "logits": logits,
        "classes": tf.argmax(logits, axis=1)
    }

    # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    #accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions["classes"])

    tf.summary.histogram("rect1_conv1_activations", rect1_normalized_conv1)
    tf.summary.histogram("rect2_conv1_activations", rect2_normalized_conv1)
    tf.summary.histogram("rect3_conv1_activations", rect3_normalized_conv1)
    # tf.summary.histogram("rect4_conv1_activations", rect4_normalized_conv1)
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    summary_hook = tf.train.SummarySaverHook(
        params["save_steps"],
        output_dir=params["logdir"],
        summary_op=tf.summary.merge_all()
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = {
            "accuracy": accuracy
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

def main():
    train_x = np.load(file_path + 'train_X.npy')
    train_y = np.load(file_path + 'train_y.npy')

    delete_indices = np.where(train_y == 0)[0][2000:]
    train_x = np.delete(train_x, delete_indices, axis=0)
    train_y = np.delete(train_y, delete_indices, axis=0)
    delete_indices = np.where(train_y == 2)[0][2000:]
    train_x = np.delete(train_x, delete_indices, axis=0)
    train_y = np.delete(train_y, delete_indices, axis=0)
    delete_indices = np.where(train_y == 3)[0][2000:]
    train_x = np.delete(train_x, delete_indices, axis=0)
    train_y = np.delete(train_y, delete_indices, axis=0)
    delete_indices = np.where(train_y == 4)[0][2000:]
    train_x = np.delete(train_x, delete_indices, axis=0)
    train_y = np.delete(train_y, delete_indices, axis=0)
    delete_indices = np.where(train_y == 5)[0][2000:]
    train_x = np.delete(train_x, delete_indices, axis=0)
    train_y = np.delete(train_y, delete_indices, axis=0)
    delete_indices = np.where(train_y == 6)[0][2000:]
    train_x = np.delete(train_x, delete_indices, axis=0)
    train_y = np.delete(train_y, delete_indices, axis=0)

    train_y = np.eye(7)[train_y]

    test_X = np.load(file_path + 'test_X.npy')
    test_y = np.load(file_path + 'test_y.npy')

    test_y = np.eye(7)[test_y]

    validation_X = np.load(file_path + 'validation_X.npy')
    validation_y = np.load(file_path + 'validation_y.npy')

    validation_y = np.eye(7)[validation_y]

    classifier = tf.estimator.Estimator(
        model_fn=model,
        model_dir=log_dir,
        params={
            "learning_rate": learning_rate,
            "momentum": momentum,
            "logdir": log_dir,
            "save_steps": save_steps
        }
    )

    evaluate_input_fn1 = tf.estimator.inputs.numpy_input_fn(
        x=test_X,
        y=test_y,
        shuffle=False,
        num_epochs=1
    )

    evaluate_input_fn2 = tf.estimator.inputs.numpy_input_fn(
        x=train_x,
        y=train_y,
        shuffle=False,
        num_epochs=1
    )

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_X,
        y=test_y,
        shuffle=False,
        num_epochs=1
    )


    classifier.train(
        input_fn=lambda : train_input_fn(train_x, train_y, batch_size),
        steps=training_steps
    )

    eval_result1 = classifier.evaluate(
        input_fn=evaluate_input_fn1
    )

    eval_result2 = classifier.evaluate(
        input_fn=evaluate_input_fn2
    )


    pred_result = classifier.predict(
        input_fn=predict_input_fn
    )

    print(list(pred_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()