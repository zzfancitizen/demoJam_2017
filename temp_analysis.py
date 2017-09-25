import numpy as np
import tensorflow as tf
import pandas as pds
import os

PATH = os.path.abspath('./data')

TRAIN_EPOCHS = 400


# feature_columns = [tf.feature_column.numeric_column("x", shape=[-1, 11])]


def my_input_fn():
    data_set = pds.read_csv(filepath_or_buffer=PATH + os.path.sep + 'temp_with_label.csv', sep='\t').as_matrix()

    X = data_set[:, 1:-2]
    Y = data_set[:, -1]

    max_value = X.max(axis=0)
    min_value = X.min(axis=0)

    # Feature scaling set input between [-1, 1]
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - min_value[i]) / (max_value[i] - min_value[i])

    return tf.estimator.inputs.numpy_input_fn(
        x={"x": X},
        y=Y,
        batch_size=50,
        num_epochs=None,
        shuffle=True)


def make_model(features, labels, mode, params, config):
    # input_layer = tf.feature_column.input_layer(
    #     features=features,
    #     feature_columns=feature_columns
    # )

    input_layer = tf.reshape(features["x"], [-1, 11])

    global_step = tf.contrib.framework.get_or_create_global_step()

    x = tf.layers.dense(
        inputs=input_layer,
        units=30,
        activation=tf.nn.relu,
        name="first_hidden_layer"
    )

    x = tf.layers.dropout(
        inputs=x,
        name="first_dropout"
    )

    x = tf.layers.dense(
        inputs=x,
        units=20,
        activation=tf.nn.relu,
        name="second_hidden_layer"
    )

    x = tf.layers.dropout(
        inputs=x,
        name="second_dropout"
    )

    x = tf.layers.dense(
        inputs=x,
        units=10,
        activation=tf.nn.relu,
        name="third_hidden_layer"
    )

    predictions = tf.contrib.layers.fully_connected(
        inputs=x,
        num_outputs=1
    )

    loss = tf.losses.absolute_difference(
        labels=labels,
        predictions=predictions
    )

    tf.summary.scalar("Loss", loss)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate,
    )

    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


def main(_):
    input_fn = my_input_fn()

    hparams = tf.contrib.training.HParams(
        learning_rate=.01,
    )

    config = tf.ConfigProto()

    trainingConfig = tf.contrib.learn.RunConfig(
        # log_device_placement=True,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        # Creates model dir (need to change this)
        model_dir=(os.path.abspath('./model') + os.path.sep + "temp-model"),
        session_config=config
    )

    estimator = tf.estimator.Estimator(
        model_fn=make_model,
        params=hparams,
        config=trainingConfig
    )

    estimator.train(
        input_fn=input_fn,
        steps=TRAIN_EPOCHS,
    )


if __name__ == '__main__':
    tf.app.run(main)
