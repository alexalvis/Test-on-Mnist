from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)#for mac
#mnist = input_data.read_data_sets('/home/workspace/python/tf/data/mnist', one_hot=True) # for windows


imgs = tf.placeholder(tf.float32, [None, 784])
yhat = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
W2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
layer1 = tf.nn.relu(tf.matmul(imgs, W1) + b1)
y = tf.matmul(layer1, W2) + b2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=yhat, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


for steps in range(50000):
    train_img_batch, train_label_batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={imgs: train_img_batch, yhat: train_label_batch})
    if steps%1000 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=yhat, logits=y)
        cross_entropy = tf.reduce_mean(loss)
        print (sess.run([accuracy, cross_entropy], feed_dict = {imgs: mnist.test.images,
                                                     yhat: mnist.test.labels}))
