from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pds
import numpy as np
import os
import random
import math

PATH = os.path.abspath('./data')
SAVE_PATH = os.path.abspath('./model/temp_simple_model')


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

    X = data_set[:5000, 1]
    x_median = np.repeat(np.median(X, axis=0), 5000, axis=0)
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    X = np.linspace(0, x_max, num=5000)
    bias = np.random.normal(0, 0.05, 5000)
    Y = np.power(abs(X - x_median), 1 / 2) * 10 / 100 + bias

    with tf.name_scope("input_layer"):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    with tf.name_scope("hidden_layer1"):
        W_h1 = tf.Variable(tf.random_normal([1, 10]), name="h1_weights")
        b_h1 = tf.Variable(tf.zeros([10]) + .1, name="h1_biases")

        h1_out = tf.nn.relu(tf.add(tf.matmul(xs, W_h1), b_h1))

    with tf.name_scope("hidden_layer2"):
        W_h2 = tf.Variable(tf.random_normal([10, 20]), name="h2_weights")
        b_h2 = tf.Variable(tf.zeros([20]) + .1, name="h2_biases")

        h2_out = tf.nn.relu(tf.add(tf.matmul(h1_out, W_h2), b_h2))

    with tf.name_scope("hidden_layer3"):
        W_h3 = tf.Variable(tf.random_normal([20, 10]), name="h3_weights")
        b_h3 = tf.Variable(tf.zeros([10]) + .1, name="h3_biases")

        h3_out = tf.nn.relu(tf.add(tf.matmul(h2_out, W_h3), b_h3))

    with tf.name_scope("output_layer"):
        W_out = tf.Variable(tf.random_normal([10, 1]), name="output_weights")
        b_out = tf.Variable(tf.zeros([1]) + .1, name="output_biases")

        logits = tf.add(tf.matmul(h3_out, W_out), b_out, name='predict')

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - logits), name="reduce_sum"), name="reduce_mean")

    train_op = tf.train.AdamOptimizer(.01).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(2000):
            # x_, y_ = next_batch(50, X, Y)

            sess.run(train_op, feed_dict={
                xs: np.reshape(X, (-1, 1)),
                ys: np.reshape(Y, (-1, 1))
            })

        save_path = saver.save(sess, SAVE_PATH + os.path.sep + 'model.ckpt')
        print("Save at %s" % save_path)


if __name__ == '__main__':
    tf.app.run(main)
