import tensorflow as tf
import os
import numpy as np

MODEL_PATH = os.path.abspath('./model/temp_simple_model')

if __name__ == '__main__':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(MODEL_PATH + os.path.sep + 'model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

        graph = tf.get_default_graph()
        xs = graph.get_tensor_by_name("input_layer/x_input:0")
        logits = graph.get_tensor_by_name("output_layer/predict:0")

        predict = sess.run(logits, feed_dict={
            xs: np.reshape([[5.]], (-1, 1))
        })

        print(predict)
