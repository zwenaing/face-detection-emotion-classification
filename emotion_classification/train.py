import numpy as np
import tensorflow as tf
# from BKKstart import cnn_start
from emotion_classification.BKVGG8 import cnn_vgg8
# from BKVGG12 import cnn_vgg12


learning_rate = 0.0005
momentum = 0.9
batch_size = 256


evaluate_batch_size = 256
num_epochs = 200
training_steps = 1000
evaluate_steps = 100
save_steps = 100

model = cnn_vgg8
file_path = '../data/fer2013/'
#log_dir = 'tmp/bkk_start/'
log_dir = "tmp/bkvgg8_batch_normalization/"
# log_dir = 'tmp/bkvgg12/'

def main():

    train_x = np.load(file_path + 'normalized_train_X.npy')
    train_y = np.load(file_path + 'train_y.npy')

    validation_X = np.load(file_path + 'normalized_validation_X.npy')
    validation_y = np.load(file_path + 'validation_y.npy')

    test_X = np.load(file_path + 'normalized_test_X.npy')
    test_y = np.load(file_path + 'test_y.npy')

    classifier = tf.estimator.Estimator(
        model_fn=model,
        model_dir=log_dir,
        params={
            "learning_rate": learning_rate,
            "momentum": momentum,
            "logdir": log_dir,
            "save_steps": save_steps
        },
        config=tf.estimator.RunConfig(log_step_count_steps=5, save_summary_steps=20)
    )

    # names = classifier.get_variable_names()
    # weights = {}
    # for name in names:
    #     weights[name] = classifier.get_variable_value(name)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        num_epochs=50,
        shuffle=True,
        queue_capacity=2000
    )

    evaluate_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=validation_X,
        y=validation_y,
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
        input_fn=train_input_fn,
        steps=training_steps
    )

    eval_result2 = classifier.evaluate(
        input_fn=evaluate_input_fn
    )

    pred_result = classifier.predict(
        input_fn=predict_input_fn
    )

    return pred_result

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()