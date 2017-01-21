#!/usr/bin/python
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


from PIL import Image
import tensorflow as tf
import numpy as np
import struct
import os
from util import *

old_cwd = os.getcwd()

def test(y_conv, y_, accuracy, h_pool4):
    final_acc = 0
    test_set.reset()
    for i in range(17):
        os.chdir(test_wd)
        batch = test_set.next_batch(1) #, decoder=tf.image.decode_jpeg)
        value = y_conv.eval(feed_dict={x: batch[0], keep_prob: 1.0})
        acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("test accuracy %g value %f" % (acc, value))
        final_acc += acc

        os.chdir(old_cwd)
        path = "activations/%d" % i
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        save_image(path + '/input', batch[0].reshape([112, 112]))
        save_activation_map(path, h_pool4.eval(feed_dict={x: batch[0]}), "Test%d" % i)

    acc = final_acc / 17.0
    print("Accuracy %f" % acc)

    return acc
    

with tf.Session() as sess:
    width, height = 112, 112
    classes = 1

    # width, height = 28, 28
    # classes = 10

    x = tf.placeholder(tf.float32, shape=[None, width * height])
    y_ = tf.placeholder(tf.float32, shape=[None, classes])

    # I'm thinking that -1 as the first dimension size indicates
    # that this dimension will be equal to that of the number of training
    # samples that are part of the selected batch
    x_image = max_pool_2x2(tf.reshape(x, [-1,width,height,1]))
    s_conv1 = [5, 5, 1, 48]
    W_conv1 = weight_variable(s_conv1) # 5 x 5 x 1 kernels 32 deep?
    b_conv1 = bias_variable([s_conv1[3]]) # A bias for each kernel neuron

    # slides filter across image, run through the ReLU activation
    # function, kernel bias is also applied
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, strides=[1, 3, 3, 1]) + b_conv1) # 36 x 36
    #h_pool1 = tf.nn.l2_normalize((h_conv1), [1, 2]) # 56 X 56
    #h_pool1 = tf.nn.l2_normalize(max_pool_2x2(h_conv1), [1, 2]) # 56 X 56
    #h_pool1 = max_pool_2x2(h_conv1) # 56 X 56
    #---------------------------------------------------------------------------
    s_conv2 = [3, 3, 48, 96]
    W_conv2 = weight_variable(s_conv2)
    b_conv2 = bias_variable([s_conv2[3]])

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    #h_pool2 = tf.nn.l2_normalize((h_conv2), [1, 2]) # 28 x 28
    h_pool2 = tf.nn.l2_normalize((h_conv2), [1, 2]) # 34 x 34
    #h_pool2 = max_pool_2x2(h_conv2) # 28 x 28
    #---------------------------------------------------------------------------
    s_conv4 = [5, 5, 96, 128]
    W_conv4 = weight_variable(s_conv4)
    b_conv4 = bias_variable([s_conv4[3]])

    h_conv4 = tf.nn.relu(conv2d(h_pool2, W_conv4, strides=[1, 3, 3, 1]) + b_conv4)
    #h_pool4 = tf.nn.l2_normalize((h_conv4), [1, 2]) # 7 x 7
    h_pool4 = tf.nn.l2_normalize((h_conv4), [1, 2]) # 10 x 10
    #h_pool4 = max_pool_2x2(h_conv4)# 7 x 7
    #h_pool4 = h_conv4# 7 x 7

    keep_prob = tf.placeholder(tf.float32)
    hp4_width, hp4_height = 10, 10 #h_conv4.shape[1] / 2, h_conv4.shape[2] / 2
    # hp4_width, hp4_height = 32, 32
    W_fc1 = weight_variable([hp4_width * hp4_height * s_conv4[3], 512])
    b_fc1 = bias_variable([512])

    h_pool4_flat = tf.reshape(h_pool4, [-1, hp4_width * hp4_height * s_conv4[3]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #W_fc2_1 = weight_variable([2048, 2048])
    #b_fc2_1 = bias_variable([2048])
    #h_fc2_1 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc2_1) + b_fc2_1)
    #h_fc2_1_drop = tf.nn.dropout(h_fc2_1, keep_prob)


    W_fc3 = weight_variable([512, 128])
    b_fc3 = bias_variable([128])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)


    W_fc4 = weight_variable([128, classes])
    b_fc4 = bias_variable([classes])

    y_conv = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    #ylna+(1−y)ln(1−a)
    
    #cost = tf.reduce_mean(-y_ * tf.log(y_conv) + (1 - y_) * tf.log(1 - y_conv))
    #cost = tf.reduce_mean(-(y_ * tf.log(y_conv)))
    exp_cost = tf.reduce_mean(tf.pow(y_conv - y_, 2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(exp_cost)
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(exp_cost)
    correct_prediction = tf.equal(y_conv / tf.abs(y_conv), y_)
    #correct_prediction = tf.less(exp_cost, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    try:
        W_conv1, b_conv1 = load_layer('conv1', w_tensor=W_conv1, b_tensor=b_conv1)
        W_conv2, b_conv2 = load_layer('conv1/conv2', w_tensor=W_conv2, b_tensor=b_conv2)
        W_conv3, b_conv3 = load_layer('conv1/conv2/conv3', w_tensor=W_conv3, b_tensor=b_conv3)
        W_conv4, b_conv4 = load_layer('conv1/conv2/conv3/conv4', w_tensor=W_conv4, b_tensor=b_conv4)
        W_fc1, b_fc1 = load_layer('conv1/conv2/conv3/conv4/fc1', w_tensor=W_fc1, b_tensor=b_fc1)
        W_fc2, b_fc2 = load_layer('conv1/conv2/conv3/conv4/fc1/fc2', w_tensor=W_fc2, b_tensor=b_fc2)
    except:
        print('Not loaded')

    print('-----------------')
    #print(b_fc3.eval())

    print("Building training set...")
    training_set = BlobTrainingSet("../data/training_blob")
    #training_set = BlobTrainingSet("../data/blob1")
    test_set = FileTrainingSet("../data/test")
    test_wd = os.getcwd()
    os.chdir(old_cwd)

    batch_size = 50
    for i in range(training_set.size() // batch_size):
      batch = training_set.next_batch(batch_size)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})
      os.write(1, bytearray('.', encoding='utf8'))

      if i%100 == 0:
        acc = test(y_conv, y_, accuracy, h_pool4)
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})

        print("\nstep %d, training accuracy %g, test accuracy %g" % (i, train_accuracy, acc))

        with open("results", "a") as text_file:
          text_file.write("Step %d Accuracy: %s\n" % (i, train_accuracy))

        if acc > 0.99:
          print('Training finished!')

    #save_layer('conv1', w_tensor=W_conv1, b_tensor=b_conv1)
    #save_layer('conv1/conv2', w_tensor=W_conv2, b_tensor=b_conv2)
    #save_layer('conv1/conv2/conv3', w_tensor=W_conv3, b_tensor=b_conv3)
    #save_layer('conv1/conv2/conv3/conv4', w_tensor=W_conv4, b_tensor=b_conv4)
    #save_layer('conv1/conv2/conv3/conv4/fc1', w_tensor=W_fc1, b_tensor=b_fc1)
    #save_layer('conv1/conv2/conv3/conv4/fc1/fc2', w_tensor=W_fc2, b_tensor=b_fc2)

    acc = test(y_conv, y_, accuracy, h_pool4)
    print('Accuracy %f' % acc)
    os.chdir(old_cwd)
