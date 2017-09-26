from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import os
import pandas as pds
import numpy as np

PATH = os.path.abspath('./data')
tf.logging.set_verbosity(v=tf.logging.ERROR)


def next_batch(batch_size, x, y):
    new_x = []
    new_y = []

    for i in range(batch_size):
        num = random.randint(0, len(x) - 1)
        new_x.append(x[num])
        new_y.append(y[num])

    return new_x, new_y


def main(_):
    data_set = pds.read_csv(
        filepath_or_buffer=PATH + os.path.sep + 'temp_with_label2.csv',
        sep='\t',
        header=None
    ).as_matrix()

    X = data_set[:, 1:-1]
    Y = data_set[:, -1]

    max_value = X.max(axis=0)
    min_value = X.min(axis=0)

    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - min_value[i]) / (max_value[i] - min_value[i])

    x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='label')

    with tf.name_scope('hidden_layer1'):
        # hidden layer 1
        W_1 = tf.Variable(tf.random_normal([11, 20]), name='h1_weight', dtype=tf.float32)
        b_1 = tf.Variable(tf.random_normal([20]), name='h1_bias', dtype=tf.float32)

        h1_o = tf.nn.sigmoid(tf.add(tf.matmul(x, W_1), b_1), name='h1_output')
    with tf.name_scope('hidden_layer2'):
        # hidden layer 2
        W_2 = tf.Variable(tf.random_normal([20, 10]), name='h2_weight', dtype=tf.float32)
        b_2 = tf.Variable(tf.random_normal([10]), name='h2_bias', dtype=tf.float32)

        h2_o = tf.nn.sigmoid(tf.add(tf.matmul(h1_o, W_2), b_2), name='h2_output')

    with tf.name_scope('output_layer'):
        W_3 = tf.Variable(tf.random_normal([10, 1]), name='output_weight', dtype=tf.float32)
        b_3 = tf.Variable(tf.random_normal([1]), name='output_bias', dtype=tf.float32)

        logits = tf.add(tf.matmul(h2_o, W_3), b_3, name='logits')

    cost = tf.reduce_max(tf.pow(logits - y, 2) / 22)

    train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss=cost)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        x_, y_ = next_batch(200, X, Y)

        sess.run(init_op)

        for _ in range(20000):
            sess.run(train_step, feed_dict={
                x: x_, y: np.reshape(y_, (-1, 1))
            })

        predict = sess.run(logits, feed_dict={
            x: [[25.0, 77.0, 24.0, 75.2, 94.0, 14.8, 9.2, 16.1, 10.0, 1013.4, 29.93]]
        })
        predict2 = sess.run(logits, feed_dict={
            x: [[9.0, 48.2, 8.0, 46.4, 93.0, 0.0, 0.0, 11.3, 7.0, 1021.2, 30.16]]
        })

        print(predict)
        print(predict2)


if __name__ == '__main__':
    tf.app.run(main)
