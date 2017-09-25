from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import pandas as pds

PATH = os.path.abspath('./data')


def main():
    data_set = pds.read_csv(filepath_or_buffer=PATH + os.path.sep + 'temp_with_label.csv', sep='\t').as_matrix()

    X = data_set[:, 1:-1]
    Y = data_set[:, -1]

    max_value = X.max(axis=0)
    min_value = X.min(axis=0)

    # Feature scaling set input between [-1, 1]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - min_value[i]) / (max_value[i] - min_value[i])

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[11])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=2,
                                                model_dir="./models/temp-model")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X, dtype=np.float32)},
        y=np.array(Y, dtype=np.int8),
        num_epochs=None,
        shuffle=True)

    # Train model.
    # classifier.fit(input_fn=train_input_fn, steps=2000)
    #
    # # Define the test inputs
    # test_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(test_set.data)},
    #     y=np.array(test_set.target),
    #     num_epochs=1,
    #     shuffle=False)
    #
    # # Evaluate accuracy.
    # accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    #
    # print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(
            np.reshape([11.0, 38.0, 11.0, 66.2, 100.0, 2.1, 26.9, 16.1, 32.0, 997.4, 19.93], (1, -1)),
            dtype=np.float32
        )},
        y=np.array(
            np.reshape([0, ], (1, -1)),
            dtype=np.int8
        ),
        num_epochs=1,
        shuffle=False)

    predict = classifier.predict(input_fn=eval_input_fn)

    print(list(predict))


if __name__ == "__main__":
    main()
