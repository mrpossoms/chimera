# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


from PIL import Image
import tensorflow as tf
import numpy as np
import struct
import os

old_cwd = os.getcwd()


class FileTrainingSet():
    def __init__(self, base_path):
        self.sample_paths = os.listdir(base_path)
        self.index = 0

        # remove hidden files
        for file in self.sample_paths:
            if file[0] == '.':
                self.sample_paths.remove(file)

        os.chdir(base_path)

    def reset(self):
        self.index = 0

    def next_batch(self, size, decoder=tf.image.decode_png):
        images, labels = [], []

        batch = self.sample_paths[self.index:self.index + size]
        self.index += size

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

        for path in batch:
            file_content = tf.read_file(path)
            print(path)
            name_and_label = path.split('.')[0].split('-')
            image = tf.image.rgb_to_grayscale(decoder(
                file_content))  # tf.to_float(decoder(file_content)) / 255.  # use png or jpg decoder based on your files.
            images += [tf.image.resize_images([image], [112, 112]).eval().flatten() / 255.0]

            if name_and_label[1] is '1':
                labels += [1.0]
            else:
                labels += [-1.0]

        # coord.request_stop()
        # coord.join(threads)

        return np.asarray(images), np.asarray(labels).reshape([len(labels), 1])


class BlobTrainingSet():
    def __init__(self, path):
        self.index = 0
        self.file = open(path, mode='rb')

    def reset(self):
        self.index = 0
        self.file.seek(0);

    def next_batch(self, size):
        images, labels = [], []

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(size):
            buf = self.file.read(112 ** 2)
            tag = struct.unpack('I', self.file.read(4))[0]

            assert tag is 0 or tag is 1

            images += [np.frombuffer(buf, dtype=np.uint8).reshape(112 ** 2) / 255.0]
            # Image.frombytes('L', (128, 128), buf).show()
            if tag == 1:
                labels += [1.0]
            else:
                labels += [-1.0]

        self.index += size
        # coord.request_stop()
        # coord.join(threads)

        return np.asarray(images), np.asarray(labels).reshape([len(labels), 1])
        # return np.asarray(images), np.asarray(labels)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    # initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable_rand(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
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


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def shape_str(np_mat):
    shape = ""
    for dim in np_mat.shape:
        shape += '-%d' % dim

    return shape


def shape_from_str(str):
    shape = []
    str = str.replace('b-', '').replace('w-', '').split('-')

    for size in str:
        shape.append(int(size))

    return shape


def load_layer(layer_path, w_tensor, b_tensor):
    weights, biases = None, None

    for file in os.listdir(layer_path):
        if file[0] is not 'b' and file[0] is not 'w':
            continue

        shape = shape_from_str(file)
        mat = np.fromfile('%s/%s' % (layer_path, file), dtype=float).reshape(shape)
        if file[0] is 'b':
            biases = mat
            print(mat)
        elif file[0] is 'w':
            weights = mat

    return w_tensor.assign(weights), b_tensor.assign(biases)


def save_layer(layer_path, w_tensor=None, b_tensor=None):
    # make sure the la
    try:
        os.mkdir(layer_path)
    except FileExistsError:
        pass

    if w_tensor is not None:
        mat = w_tensor.eval().astype(float)
        mat.tofile('%s/w%s' % (layer_path, shape_str(mat)))

    if b_tensor is not None:
        mat = b_tensor.eval().astype(float)
        mat.tofile('%s/b%s' % (layer_path, shape_str(mat)))


def save_image(path, array):
    bitmap = (array * 255).astype(np.uint8)
    img = Image.fromarray(bitmap, mode='L')
    img.save(path + '.png')


def save_activation_map(path, tensor, name):
    act_maps = np.dsplit(tensor[0], tensor.shape[3])
    filter_idx = 0
    for act_map in act_maps:
        save_image(path + '/' + str(filter_idx), act_map.reshape(act_map.shape[:2]))
        filter_idx += 1


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


def generate_layer(last, genotype):
    last_shape, last_layer, last_genotype = last

    is_conv = not (genotype & 0x01)
    genes = genotype >> 1
    shape = last_shape.copy()
    input = last_layer

    if is_conv:
        side = [3, 5, 7][genes & 0x03]
        depth = 2 ** ((genes >> 2) & 0x07)
        normalize = bool((genes >> 5) & 0x01)
        pooling = bool(genes >> 6)

        if len(last_shape) < 3:
            k = int(len(last_shape.flatten()) ** 0.5)
            last_shape = [k, k, 1, depth]
            shape = last_shape.copy()
            input = input.reshape([-1, k, k, 1])

        shape[2] = depth
        weights = weight_variable([side, side, last_shape[2], shape[2]])
        biases = bias_variable([depth])
        layer = tf.nn.relu(conv2d(input, weights) + biases)

        if normalize:
            layer = tf.nn.l2_normalize(layer, [1, 2])

        if pooling:
            layer = max_pool_2x2(layer)
            shape[0] >>= 1
            shape[1] >>= 1

        return shape, layer, genotype

    else:
        neurons = 2 ** (genes & 0x0F)
        keep_prob = ((genes >> 4) + 1) / 8.0
        last_neurons = np.prod(last_shape)

        if len(last_shape) > 1:
            input = tf.reshape(input, [-1, last_neurons])

        shape = [last_neurons, neurons]
        weights = weight_variable(shape)
        biases = bias_variable_rand([neurons])
        layer = tf.nn.dropout(tf.nn.relu(tf.matmul(input, weights) + biases), keep_prob)

        return shape, layer, genotype


with tf.Session() as sess:
    width, height = 112, 112
    classes = 1

    # width, height = 28, 28
    # classes = 10

    genome = [0x9A, 0x92, 0x92, 0x77]

    x = tf.placeholder(tf.float32, shape=[None, width * height])
    y_ = tf.placeholder(tf.float32, shape=[None, classes])

    # I'm thinking that -1 as the first dimension size indicates
    # that this dimension will be equal to that of the number of training
    # samples that are part of the selected batch
    x_image = tf.reshape(x, [-1, width, height, 1])

    last = [width, height, 1], x_image, 0

    for gene in genome:
        last = generate_layer(last, gene)

    # s_conv1 = [5, 5, 1, 8]
    # W_conv1 = weight_variable(s_conv1) # 5 x 5 x 1 kernels 32 deep?
    # b_conv1 = bias_variable([s_conv1[3]]) # A bias for each kernel neuron
    #
    # # slides filter across image, run through the ReLU activation
    # # function, kernel bias is also applied
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # #h_pool1 = tf.nn.l2_normalize((h_conv1), [1, 2]) # 56 X 56
    # #h_pool1 = tf.nn.l2_normalize(max_pool_2x2(h_conv1), [1, 2]) # 56 X 56
    # h_pool1 = max_pool_4x4(h_conv1) # 56 X 56
    # ##---------------------------------------------------------------------------
    # s_conv2 = [5, 5, 8, 4]
    # W_conv2 = weight_variable(s_conv2)
    # b_conv2 = bias_variable([s_conv2[3]])
    #
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    # #h_pool2 = tf.nn.l2_normalize((h_conv2), [1, 2]) # 28 x 28
    # #h_pool2 = tf.nn.l2_normalize(max_pool_2x2(h_conv2), [1, 2]) # 28 x 28
    # #---------------------------------------------------------------------------
    # s_conv_f = [5, 5, 4, 4]
    # W_conv_f = weight_variable(s_conv_f)
    # b_conv_f = bias_variable([s_conv_f[3]])
    #
    # h_conv_f = tf.nn.relu(conv2d(h_pool2, W_conv_f) + b_conv_f)
    # #h_pool2 = tf.nn.l2_normalize((h_conv2), [1, 2]) # 28 x 28
    # #h_pool_f = tf.nn.l2_normalize(max_pool_2x2(h_conv_f), [1, 2]) # 28 x 28
    # h_pool_f = max_pool_2x2(h_conv_f) # 28 x 28
    # #---------------------------------------------------------------------------
    keep_prob = tf.placeholder(tf.float32)
    # hp4_width, hp4_height = 7, 7 #h_conv4.shape[1] / 2, h_conv4.shape[2] / 2
    # # hp4_width, hp4_height = 32, 32
    last_hl_neurons = last[0][1]
    if len(last[0]) > 2:
        last_hl_neurons = np.prod(last[0])

    h_pool_f_flat = tf.reshape(last[1], [-1, last_hl_neurons])

    W_fc1 = weight_variable([last_hl_neurons, classes])
    b_fc1 = bias_variable_rand([classes])
    # #y_conv = tf.nn.relu(tf.matmul(h_pool_f_flat, W_fc1) + b_fc1)
    y_conv = tf.matmul(h_pool_f_flat, W_fc1) + b_fc1

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    # ylna+(1−y)ln(1−a)

    # cost = tf.reduce_mean(-y_ * tf.log(y_conv) + (1 - y_) * tf.log(1 - y_conv))
    # cost = tf.reduce_mean(-(y_ * tf.log(y_conv)))
    exp_cost = tf.reduce_mean(tf.pow(y_conv - y_, 2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(exp_cost)
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(exp_cost)
    correct_prediction = tf.equal(y_conv / tf.abs(y_conv), y_ / tf.abs(y_))
    # correct_prediction = tf.less(exp_cost, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    try:
        pass
        # W_conv1, b_conv1 = load_layer('conv1', w_tensor=W_conv1, b_tensor=b_conv1)
        # W_conv2, b_conv2 = load_layer('conv1/conv2', w_tensor=W_conv2, b_tensor=b_conv2)
        # W_conv3, b_conv3 = load_layer('conv1/conv2/conv3', w_tensor=W_conv3, b_tensor=b_conv3)
        # W_conv4, b_conv4 = load_layer('conv1/conv2/conv3/conv4', w_tensor=W_conv4, b_tensor=b_conv4)
        # W_fc1, b_fc1 = load_layer('conv1/conv2/conv3/conv4/fc1', w_tensor=W_fc1, b_tensor=b_fc1)
        # W_fc2, b_fc2 = load_layer('conv1/conv2/conv3/conv4/fc1/fc2', w_tensor=W_fc2, b_tensor=b_fc2)
    except:
        print('Not loaded')

    print('-----------------')
    # print(b_fc2.eval())

    print("Building training set...")
    training_set = BlobTrainingSet("../data/training_blob")
    # training_set = BlobTrainingSet("../data/blob1")
    test_set = FileTrainingSet("../data/test")
    test_wd = os.getcwd()
    os.chdir(old_cwd)

    for i in range(2000):
        batch = training_set.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})
        os.write(1, bytearray('.', encoding='utf8'))

        if i % 500 == 0:
            test(y_conv, y_, accuracy, None)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})

            print("\nstep %d, training accuracy %g" % (i, train_accuracy))

            with open("results", "a") as text_file:
                text_file.write("Step %d Accuracy: %s\n" % (i, train_accuracy))

    test(y_conv, y_, accuracy, None)
    # save_layer('conv1', w_tensor=W_conv1, b_tensor=b_conv1)
    # save_layer('conv1/conv2', w_tensor=W_conv2, b_tensor=b_conv2)
    # save_layer('conv1/conv2/conv3', w_tensor=W_conv3, b_tensor=b_conv3)
    # save_layer('conv1/conv2/conv3/conv4', w_tensor=W_conv4, b_tensor=b_conv4)
    # save_layer('conv1/conv2/conv3/conv4/fc1', w_tensor=W_fc1, b_tensor=b_fc1)
    # save_layer('conv1/conv2/conv3/conv4/fc1/fc2', w_tensor=W_fc2, b_tensor=b_fc2)

    os.chdir(old_cwd)
