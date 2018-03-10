import tensorflow as tf
import numpy as np

learning_rate = 0.01
batch_size =256
training_steps = 1000
file_path = '../data/fer2013/'

def cnn_model(features, labels, mode):
    input = tf.cast(tf.reshape(features, [-1, 42, 42, 1]), tf.float32)
    conv1 = tf.layers.conv2d(input, filters=32, kernel_size=[5, 5], strides=1, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=2)
    conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=[4, 4], strides=1, activation=tf.nn.relu)
    pool2 = tf.layers.average_pooling2d(conv2, pool_size=[3, 3], strides=2)
    conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=[5, 5], strides=1, activation=tf.nn.relu)
    pool3 = tf.layers.average_pooling2d(conv3, pool_size=[3, 3], strides=2)
    flatten_pool3 = tf.reshape(pool3, [-1, 16 * 16 * 64])
    fc1 = tf.layers.dense(flatten_pool3, units=3072, activation=tf.nn.relu)
    fc1 = tf.layers.dropout(fc1, rate=0.1)
    logits = tf.layers.dense(fc1, units=7)

    probabilities = tf.nn.softmax(logits, axis=1)
    y_pred = tf.argmax(probabilities, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "probabilities": probabilities,
            "logits": logits,
            "predictions": y_pred
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy =  tf.metrics.accuracy(labels=labels, predictions=y_pred)
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
    train_X = train_X[:, :1764]
    print(train_X.shape)

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

