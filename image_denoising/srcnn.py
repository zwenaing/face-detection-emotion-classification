import tensorflow as tf
from image_denoising.srcnn_input import load_data

BATCH_SIZE = 128
NUM_STEPS = 1000
LEARNING_RATE = 0.001


def srcnn_model(features, labels, mode, params):
    with tf.name_scope('input_layer'):
        input_data = tf.cast(features['images'], dtype=tf.float32)

    with tf.name_scope('hidden_1'):
        input_layer = tf.layers.conv2d(
            inputs=input_data,
            kernel_size=[9, 9],
            filters=64,
            padding='valid',
            strides=1,
            activation=tf.nn.relu
        )

    with tf.name_scope('hidden_2'):
        conv_1 = tf.layers.conv2d(
            inputs=input_layer,
            kernel_size=[1, 1],
            filters=32,
            strides=1,
            padding='valid',
            activation=tf.nn.relu
        )

    with tf.name_scope('output_layer'):
        conv_2 = tf.layers.conv2d(
            inputs=conv_1,
            kernel_size=[5, 5],
            filters=1,
            strides=1,
            padding='valid',
            activation=None
        )

    with tf.name_scope('output_reshape'):
        logits = tf.reshape(conv_2, [-1, 441])

    with tf.name_scope('label_reshape'):
        labels = tf.reshape(labels, [-1, 441])

    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    with tf.name_scope('training'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    with tf.name_scope('train_accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(logits, labels)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('evaluation'):
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(labels=labels, predictions=logits)
            eval_metrics = {'accuracy': accuracy}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('checkpoints/train')
    test_writer = tf.summary.FileWriter('checkpoints/test')

def main():

    (train_x, train_y), (test_x, test_y) = load_data()

    model = tf.estimator.Estimator(
        model_fn=srcnn_model,
        model_dir='checkpoints',
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": train_x},
        y=train_y,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True
    )

    model.train(
        input_fn=train_input_fn,
        steps=NUM_STEPS
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": test_x},
        y=test_y,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    eval_result = model.evaluate(
        input_fn=eval_input_fn,
        steps=1
    )
    print(eval_result)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

