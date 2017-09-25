from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pds
import os

# tf.logging.set_verbosity(tf.logging.INFO)

PATH = os.path.abspath('./data')


def my_input_fn():
    data_set = pds.read_csv(filepath_or_buffer=PATH + os.path.sep + 'temp_with_label.csv', sep='\t').as_matrix()

    X = data_set[:, 1:-2]
    Y = data_set[:, -1]

    max_value = X.max(axis=0)
    min_value = X.min(axis=0)

    # Feature scaling set input between [-1, 1]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - min_value[i]) / (max_value[i] - min_value[i])

    print(X.shape, Y.shape)

    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X)},
        y=np.array(Y),
        batch_size=50,
        num_epochs=None,
        shuffle=True)


def mode_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 10])

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

    prediction = tf.contrib.layers.fully_connected(inputs=hid3,
                                                   num_outputs=1)

    loss = tf.losses.absolute_difference(labels=labels,
                                         predictions=prediction)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=.1
    )

    train_op = optimizer.minimize(loss=loss,
                                  global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op)


def main(_):
    input_fn = my_input_fn()

    classifier = tf.estimator.Estimator(
        model_fn=mode_fn, model_dir='./model/temp-model-1'
    )

    classifier.train(
        input_fn=input_fn,
        steps=2000
    )


if __name__ == "__main__":
    tf.app.run(main)
