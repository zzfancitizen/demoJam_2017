from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import pandas as pds

PATH = os.path.abspath('./data')

tf.logging.set_verbosity(tf.logging.ERROR)


def main(_):
    data_set = pds.read_csv(
        filepath_or_buffer=PATH + os.path.sep + 'temp_with_label2.csv',
        sep='\t'
    ).as_matrix()

    X = data_set[:, 1:-1]
    Y = data_set[:, -1]

    # print(X[0], Y[0])
    # print(X.shape, Y.shape)

    # max_value = X.max(axis=0)
    # min_value = X.min(axis=0)

    # for i in range(X.shape[1]):
    #     X[:, i] = (X[:, i] - min_value[i]) / (max_value[i] - min_value[i])

    feature_columns = [tf.feature_column.numeric_column("x", shape=[11])]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[20, 10],
        optimizer=tf.train.AdamOptimizer(
            learning_rate=1,
        ),
        model_dir='./model/temp-model-2'
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X, dtype=np.float32)},
        y=np.array(np.reshape(Y, (-1, 1)), dtype=np.float32),
        batch_size=50,
        num_epochs=None,
        shuffle=True
    )

    regressor.fit(input_fn=train_input_fn, steps=2000)
    # regressor.fit(
    #     x=np.array(X, dtype=np.float32),
    #     y=np.array(np.reshape(Y, (-1, 1)), dtype=np.float32),
    #     batch_size=50,
    #     steps=2000
    # )


if __name__ == "__main__":
    tf.app.run(main)
