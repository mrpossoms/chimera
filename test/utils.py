from PIL import Image
import tensorflow as tf
import numpy as np
import os
import random
import struct


class Zip:
    def __init__(self, *lists):
        self.lists = []
        self.longest = 0
        self.i = 0

        for list in lists:
            self.longest = len(list) if len(list) > self.longest else self.longest
            self.lists.append(list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.longest:
            raise StopIteration
        else:
            tuple = []
            for list in self.lists:
                tuple.append(list[self.i] if len(list) > self.i else None)

            self.i += 1

            return tuple


class FileTrainingSet():
    def __init__(self, base_path):
        self.sample_paths = os.listdir(base_path)
        self.index = 0
        self.cache = {}

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
            image = None
            name_and_label = path.split('.')[0].split('-')

            if path in self.cache:
                image = self.cache[path]
            else:
                file_content = tf.read_file(path)
                print(path)
                grey = tf.image.rgb_to_grayscale(decoder(
                    file_content))  # tf.to_float(decoder(file_content)) / 255.  # use png or jpg decoder based on your files.
                image = tf.image.resize_images([grey], [112, 112]).eval().flatten() / 255.0
                self.cache[path] = image

            images += [image]

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


def pick(n, from_list=[None]):
    chosen = []
    for i in range(n):
        chosen += [from_list[random.randint(0, len(from_list) - 1)]]

    return chosen

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
