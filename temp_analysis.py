from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pds
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

PATH = os.path.abspath('./data')


def my_input_fn():
    data_set = pds.read_csv(filepath_or_buffer=PATH + os.path.sep + 'temp_with_label.csv', sep='\t').as_matrix()

    X = data_set[:, 1:-1]
    Y = data_set[:, -1]

    # print(X[0], Y[0])

    max_value = X.max(axis=0)
    min_value = X.min(axis=0)

    # Feature scaling set input between [-1, 1]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - min_value[i]) / (max_value[i] - min_value[i])

    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X, dtype=np.float32)},
        y=np.array(np.reshape(Y, (-1, 1)), dtype=np.float32),
        batch_size=50,
        num_epochs=None,
        shuffle=True)


def mode_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 11])

    hid1 = tf.layers.dense(inputs=input_layer,
                           units=10,
                           activation=tf.nn.relu,
                           name='hidden_layer_1'
                           )

    drop1 = tf.layers.dropout(inputs=hid1, rate=0.4, name='dropout_1')

    hid2 = tf.layers.dense(inputs=drop1,
                           units=20,
                           activation=tf.nn.relu,
                           name='hidden_layer_2'
                           )

    hid3 = tf.layers.dense(inputs=hid2,
                           units=10,
                           activation=tf.nn.relu,
                           name='hidden_layer_3')

    logits = tf.layers.dense(inputs=hid3,
                             units=1)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=.5
        )

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(_):
    input_fn = my_input_fn()

    classifier = tf.estimator.Estimator(
        model_fn=mode_fn, model_dir='./model/temp-model-1'
    )

    classifier.train(
        input_fn=input_fn,
        steps=2000
    )

    # _X = np.reshape([23.3, 73.9, 21.7, 71.1, 91.0, 42.6, 26.5, 3.2, 2.0, 1005.0, 29.68], (1, -1))
    # _Y = np.reshape([0, ], (1, -1))
    #
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(
    #         _X,
    #         dtype=np.float32
    #     )},
    #     y=np.array(
    #         _Y,
    #         dtype=np.int8
    #     ),
    #     num_epochs=1,
    #     shuffle=False)
    #
    # eval_results = classifier.evaluate(input_fn=eval_input_fn)
    # # predict = classifier.predict(input_fn=eval_input_fn)
    # print(eval_results)
    # # print(predict)
    # # print(list(predict))


if __name__ == "__main__":
    tf.app.run(main)
