
from PIL import Image
import tensorflow as tf
import numpy as np
import os

ever=0

class TrainingSet():
    def __init__(self, base_path):
        self.sample_paths = os.listdir(base_path)
        self.index = 0

        os.chdir(base_path)

    def next_batch(self, size):
        images, labels = [], []

        batch = self.sample_paths[self.index:self.index + size]
        self.index += size

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

        for path in batch:
            file_content = tf.read_file(path)
            print(path)
            name_and_label = path.split('.')[0].split('-')
            image = tf.to_float(tf.image.decode_png(file_content)) / 255.  # use png or jpg decoder based on your files.
            images += [image.eval().flatten()]

            if name_and_label[1] is '1':
                labels += [np.array([1, 0])]
            else:
                labels += [np.array([0, 1])]

        # coord.request_stop()
        # coord.join(threads)

        return np.asarray(images), np.asarray(labels).reshape(50, 2)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
    # strides
    #   [0] - batch
    #   [1] - width
    #   [2] - height
    #   [3] - channels
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  # max pooling here is sliding a 2x2 filter across the activation
  # map, and selecting the max activation as the value to be used
  # for the downsampled pool. This effectively divides the resolution
  # of the activation map by two
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


print("Building training set...")
training_set = TrainingSet("../data")

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[None, 128**2])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # I'm thinking that -1 as the first dimension size indicates
    # that this dimension will be equal to that of the number of training
    # samples that are part of the selected batch
    x_image = tf.reshape(x, [-1,128,128,1])
    W_conv1 = weight_variable([5, 5, 1, 32]) # 5 x 5 x 1 kernels 32 deep?
    b_conv1 = bias_variable([32]) # A bias for each kernel neuron

    # slides filter across image, run through the ReLU activation
    # function, kernel bias is also applied
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 64 x 64
    #---------------------------------------------------------------------------
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # 32 x 32
    #---------------------------------------------------------------------------
    W_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3) # 16 x 16
    #---------------------------------------------------------------------------
    W_conv4 = weight_variable([5, 5, 64, 64])
    b_conv4 = bias_variable([64])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4) # 8 x 8


    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool4_flat = tf.reshape(h_pool4, [-1, 8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(20000):
      batch = training_set.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # print("test accuracy %g"%accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
