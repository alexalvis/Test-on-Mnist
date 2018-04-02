from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"] , [-1, 28, 28, 1])
    #print(input_layer.shape)
    #print (features["x"].shape)
    conv1 = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu )
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    #conv3 = tf.layers.conv2d(inputs = pool2, filters = 128, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
    #pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2, 2], strides = 2)
    #pool3_flat = tf.reshape(pool3, [-1, 7* 7* 128])
    dense = tf.layers.dense(inputs = pool2_flat, units= 1024, activation= tf.nn.relu)
    dropout = tf.layers.dropout(inputs = dense, rate = 0.5, training= mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs = dropout, units= 10)
    predictions = {
        "classes": tf.argmax(input = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.05)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def main(unused_argv):
    #train_date = np.load("mnist_train_images.npy")
    #train_labels = np.load("mnist_train_labels.npy")
    #eval_date = np.load("mnist_test_images.npy")
    #eval_labels = np.load("mnist_test_labels.npy")

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    #print(train_data.shape)
    mnist_classifier = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir = "tmp/mnist_convnet_model")
    #tensors_to_log = {"softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter= 100)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = 200,
        num_epochs = None,
        shuffle = True
    )  #define input
    mnist_classifier.train(input_fn = train_input_fn, steps = 2000#, hooks = [logging_hook]
    )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)
if __name__ == "__main__":
    tf.app.run()