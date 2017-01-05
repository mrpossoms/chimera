# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


from PIL import Image
import tensorflow as tf
import numpy as np
import os
import random
from utils import *

old_cwd = os.getcwd()


def test(y_conv, y_, accuracy, h_pool4):
    final_acc = 0
    test_set.reset()
    for i in range(14):
        os.chdir(test_wd)
        batch = test_set.next_batch(1)  # , decoder=tf.image.decode_jpeg)
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

        if h_pool4 is not None:
            save_activation_map(path, h_pool4.eval(feed_dict={x: batch[0]}), "Test%d" % i)

    print("Accuracy %f" % (final_acc / 13.0))


def breed(left, right):
    offspring = []

    for left_gene, right_gene in Zip(left, right):
        gene = left_gene

        if random.randint(0, 1):
            gene = right_gene

        # give a 1% chance that the gene will be duplicated
        if random.random() < 0.01 and gene is not None:
            offspring += [gene, gene]
        else:
            offspring += [gene]

    return offspring


def random_genome(length):
    genome = []
    for _ in range(length):
        genome += [random.randint(0, 255)]

    return genome


def mutate(genome, mutation_frequency=0.1):
    for i in range(len(genome)):
        mut_mask = 0
        for bit in range(8):
            if random.random() < mutation_frequency:
                mut_mask += 1 << bit

        genome[i] ^= mut_mask


def populate(candidates, max=100):
    population = candidates.copy()

    while len(population) < max:
        left, right = pick(2, from_list=candidates)

        offspring = breed(left, right)
        mutate(offspring, random.random() * 0.1)

        population += [offspring]

    return population


def generate_layer(last, genotype):
    last_shape, last_layer, last_genotype = last
    last_is_conv = not (last_genotype & 0x01)

    is_conv = not (genotype & 0x01) and last_is_conv is True
    genes = genotype >> 1
    shape = last_shape.copy()
    input = last_layer

    if is_conv:
        side = [3, 5, 7][genes & 0x02]
        depth = 2 ** ((genes >> 2) & 0x07)
        normalize = bool((genes >> 5) & 0x01)
        pooling = bool(genes >> 6)

        shape[2] = depth
        weights = weight_variable([side, side, last_shape[2], shape[2]])
        biases = bias_variable([depth])
        layer = tf.nn.relu(conv2d(input, weights) + biases)

        if normalize:
            layer = tf.nn.l2_normalize(layer, [1, 2])

        if pooling and shape[0] > 2 and shape[1] > 2:
            layer = max_pool_2x2(layer)
            shape[0] >>= 1
            shape[1] >>= 1

        if len(shape) is not 3:
            raise LookupError

        return shape, layer, genotype

    else:
        neurons = 2 ** (genes & 0x0F)
        keep_prob = ((genes >> 4) + 1) / 8.0
        last_neurons = last_shape[len(last_shape) - 1]

        if last_is_conv:
            last_neurons = np.prod(last_shape)
            input = tf.reshape(input, [-1, last_neurons])

        shape = [last_neurons, neurons]
        weights = weight_variable(shape)
        biases = bias_variable_rand([neurons])
        layer = tf.nn.dropout(tf.nn.relu(tf.matmul(input, weights) + biases), keep_prob)

        return shape, layer, genotype


genome = [0x9A, 0x92, 0x92, 0x77]

starters = [genome]

for _ in range(10):
    starters += [random_genome(4)]

training_set = BlobTrainingSet("../data/training_blob")
test_set = FileTrainingSet("../data/test")

while True:
    population = populate(starters)
    evaluations = []

    for candidate in population:
        width, height = 112, 112
        classes = 1

        # width, height = 28, 28
        # classes = 10

        sess = tf.Session()
        with sess.as_default():
            training_set.reset()
            test_set.reset()

            keep_prob = tf.placeholder(tf.float32)
            x = tf.placeholder(tf.float32, shape=[None, width * height])
            y_ = tf.placeholder(tf.float32, shape=[None, classes])

            # I'm thinking that -1 as the first dimension size indicates
            # that this dimension will be equal to that of the number of training
            # samples that are part of the selected batch
            x_image = tf.reshape(x, [-1, width, height, 1])

            last = [width, height, 1], x_image, 0

            for gene in candidate:
                last = generate_layer(last, gene)

            last_hl_neurons = last[0][1]
            if len(last[0]) > 2:
                last_hl_neurons = np.prod(last[0])

            h_pool_f_flat = tf.reshape(last[1], [-1, last_hl_neurons])

            W_fc1 = weight_variable([last_hl_neurons, classes])
            b_fc1 = bias_variable_rand([classes])
            y_conv = tf.matmul(h_pool_f_flat, W_fc1) + b_fc1

            exp_cost = tf.reduce_mean(tf.pow(y_conv - y_, 2))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(exp_cost)
            correct_prediction = tf.equal(y_conv / tf.abs(y_conv), y_ / tf.abs(y_))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            sess.run(tf.global_variables_initializer())

            test_wd = os.getcwd()
            os.chdir(old_cwd)

            for i in range(100):
                batch = training_set.next_batch(50)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})
                os.write(1, bytearray('.', encoding='utf8'))

                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})

                    print("\nstep %d, training accuracy %g" % (i, train_accuracy))

                    with open("results", "a") as text_file:
                        text_file.write("Step %d Accuracy: %s\n" % (i, train_accuracy))

            test(y_conv, y_, accuracy, None)

            os.chdir(old_cwd)
        sess.close()
