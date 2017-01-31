#!/usr/bin/python
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


from PIL import Image
import tensorflow as tf
import numpy as np
import struct
import os
from util import *
from fibers import *

old_cwd = os.getcwd()

def test(y_conv, y_, accuracy, h_pool4):
    final_acc = 0
    test_set.reset()
    for i in range(test_set.size()):
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

        #save_image(path + '/input', batch[0].reshape([112, 112]))
        
        if h_pool4:
            save_activation_map(path, h_pool4.eval(feed_dict={x: batch[0]}), "Test%d" % i)

    acc = final_acc / test_set.size()
    print("Accuracy %f" % acc)

    return acc
    

with tf.Session() as sess:
    width, height, depth = 112, 112, 3
    classes = 1

    # width, height = 28, 28
    # classes = 10

    x = tf.placeholder(tf.float32, shape=[None, width * height * depth])
    y_ = tf.placeholder(tf.float32, shape=[None, classes])

    # I'm thinking that -1 as the first dimension size indicates
    # that this dimension will be equal to that of the number of training
    # samples that are part of the selected batch
    x_image = ((max_pool_2x2(tf.reshape(x, [-1,width,height,depth]))))
    keep_prob = tf.placeholder(tf.float32)

    y_conv = (input(x_image, [56,56,3])
                .to_conv(filter=[3,3,96], stride=[2, 2, 1])
                .to_conv(filter=[3,3,96])
                .out(1)
            ) - 1

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

    print("Building training set...")
    training_set = BlobTrainingSet("../data/training_blob", shape=[112, 112, 3])
    #training_set = BlobTrainingSet("../data/blob1")
    test_set = FileTrainingSet("../data/test", shape=[112, 112, 3])
    test_wd = os.getcwd()
    os.chdir(old_cwd)

    batch_size = 50
    for i in range(training_set.size() // batch_size):
      batch = training_set.next_batch(batch_size)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})
      #os.write(1, bytearray('.', encoding='utf8'))

      if i%100 == 0:
        acc = test(y_conv, y_, accuracy, None)
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})

        print("\nstep %d, training accuracy %g, test accuracy %g" % (i, train_accuracy, acc))

        with open("results", "a") as text_file:
          text_file.write("Step %d Accuracy: %s\n" % (i, train_accuracy))

        if acc > 0.99:
          print('Training finished!')

    acc = test(y_conv, y_, accuracy, None)
    print('Accuracy %f' % acc)
    os.chdir(old_cwd)
