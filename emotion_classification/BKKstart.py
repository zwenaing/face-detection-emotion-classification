from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""
Method for building BKKstart CNN architecture. 
"""
def cnn_start(features, labels, mode, params):
    input = tf.cast(tf.reshape(features, [-1, 42, 42, 1]), tf.float32)

    conv1 = tf.layers.conv2d(
        input,
        filters=32,
        kernel_size=[5, 5],
        strides=1,
        activation=tf.nn.relu,
        padding="same",
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=[3, 3],
        strides=2
    )
    conv2 = tf.layers.conv2d(
        pool1,
        filters=32,
        kernel_size=[4, 4],
        strides=1,
        activation=tf.nn.relu,
        padding="same",
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )
    pool2 = tf.layers.average_pooling2d(
        conv2,
        pool_size=[3, 3],
        strides=2
    )
    conv3 = tf.layers.conv2d(
        pool2,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        activation=tf.nn.relu,
        padding="same",
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )
    pool3 = tf.layers.average_pooling2d(
        conv3,
        pool_size=[3, 3],
        strides=2
    )

    flatten_pool3 = tf.reshape(pool3, [-1, 4 * 4 * 64])

    fc1 = tf.layers.dense(flatten_pool3, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(fc1, rate=0.5, training= mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(dropout, units=7, activation=tf.nn.softmax)

    predictions = {
        "probabilities": tf.nn.softmax(logits, axis=1),
        "logits": logits,
        "classes": tf.argmax(logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = {
            "accuracy": accuracy
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)

    tf.summary.histogram("conv1_activations", conv1)
    tf.summary.histogram("conv2_activations", conv2)
    tf.summary.histogram("conv3_activations", conv3)
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    summary_hook = tf.train.SummarySaverHook(
        params["save_steps"],
        output_dir=params["logdir"],
        summary_op=tf.summary.merge_all()
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=params["learning_rate"], momentum=params["momentum"])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])