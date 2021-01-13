"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""
import os
import pickle
import tensorflow as tf

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.io import imshow


def build_loader(path, dataset, transform_center=False):
    batch_size = 1
    test_example_num = None
    if dataset == 'svhn':
        dloader = SVHNData(path)
        test_example_num = 26032
    elif dataset == 'fmnist':
        dloader = FMNISTData(path)
        test_example_num = 10000
    elif dataset == 'cifar10':
        dloader = CIFAR10Data(path)
        test_example_num = 10000
    elif dataset == 'celeb5':
        dloader = CELEB5Data(path)
    elif dataset == 'celebA':
        dloader = CELEBAData(path)
    else:
        print('dataset {} is not supported:'.format(dataset))
        raise

    if transform_center:
        with tf.variable_scope('input'):
            images = tf.placeholder(tf.float32, shape=[None, dloader.eval_data.xs.shape[1], dloader.eval_data.xs.shape[2], dloader.eval_data.xs.shape[3]])
        mode = 'rot_and_shift'
        angle = 30
        x_shift = 3
        y_shift = 3
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, visible_device_list='0')
        _transformed_images = transform(images, mode=mode, angle=angle, x_shift=x_shift, y_shift=y_shift, standardize=False)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for left in range(0, test_example_num, batch_size):
                right = min(left+batch_size, test_example_num)
                transformed_images = sess.run([_transformed_images], feed_dict={images: dloader.eval_data.xs[left:right]})
                dloader.eval_data.xs[left:right] = transformed_images

    # from utils_new import visualize_imgs
    # tmp_img_folder = 'tmp_img'
    # if not os.path.exists(tmp_img_folder):
    #     os.mkdir(tmp_img_folder)
    # print(dloader.eval_data.xs[0])
    # visualize_imgs(tmp_img_folder+'/', [dloader.eval_data.xs[:2]/255], 'tmp')

    return dloader

class CELEB5Data(object):
    def __init__(self, path):
        train_images = np.load(os.path.join(path, 'train_x.npy'))
        train_labels = np.load(os.path.join(path, 'train_y.npy'))
        eval_images = np.load(os.path.join(path, 'test_x.npy'))
        eval_labels = np.load(os.path.join(path, 'test_y.npy'))

        print(train_images.shape, eval_images.shape)

        train_images = resize(train_images, (91, 32, 32, 3), anti_aliasing=True)
        eval_images = resize(eval_images, (24, 32, 32, 3), anti_aliasing=True)
        # print(train_labels)
        # imshow(train_images[5])
        # plt.show()
        # print(train_images.shape, eval_images.shape)

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

class CELEBAData(object):
    def __init__(self, path):
        train_images = np.load(os.path.join(path, 'train_x.npy'))
        train_labels = np.load(os.path.join(path, 'train_y.npy'))
        eval_images = np.load(os.path.join(path, 'test_x.npy'))
        eval_labels = np.load(os.path.join(path, 'test_y.npy'))

        print(train_images.shape, eval_images.shape)

        train_images = resize(train_images, (2499, 32, 32, 3), anti_aliasing=True)
        eval_images = resize(eval_images, (500, 32, 32, 3), anti_aliasing=True)
        # print(train_labels)
        # imshow(train_images[5])
        # plt.show()
        # print(train_images.shape, eval_images.shape)

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

class SVHNData(object):
    def __init__(self, path):
        train_filename = os.path.join(path, 'train_32x32.mat')
        eval_filename = os.path.join(path, 'test_32x32.mat')

        train_images, train_labels = self._load_datafile(train_filename)
        eval_images, eval_labels = self._load_datafile(eval_filename)

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

    @staticmethod
    def _load_datafile(filename):
        d = scipy.io.loadmat(filename)
        images = d['X'].transpose(3, 0, 1, 2)
        labels = d['y'].squeeze()
        # labels need to be 0-indexed
        labels -= 1
        return images, labels

class FMNISTData(object):
    def __init__(self, path):
        train_images = np.load(os.path.join(path, 'train_images.npy'))
        train_labels = np.load(os.path.join(path, 'train_labels.npy'))
        eval_images = np.load(os.path.join(path, 'test_images.npy'))
        eval_labels = np.load(os.path.join(path, 'test_labels.npy'))

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

class CIFAR10Data(object):
    def __init__(self, path):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
            train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        with open(os.path.join(path, metadata_filename), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])





class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.arange(self.n)


    def get_next_batch(self, batch_size, multiple_passes=True, reshuffle_after_pass=False):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        batch_end = self.batch_start + actual_batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        if batch_end == self.n:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        else:
            self.batch_start = batch_end

        return batch_xs, batch_ys

    def get_next_batch_reweight(self, batch_size, train_neighbor_avg_acc=np.nan, exp_num=1):
        sample_prob = np.ones(self.n) / self.n

        if type(train_neighbor_avg_acc) is np.ndarray:
            # sample_prob = (1-train_neighbor_avg_acc)*100
            sample_prob = ((1-train_neighbor_avg_acc)**exp_num) * (100**exp_num)
            sample_prob = (np.ones_like(sample_prob)+sample_prob) / (100**exp_num + 1)
            sample_prob = sample_prob / np.sum(sample_prob)

        chosen_inds = np.random.choice(self.n, size=batch_size, replace=False, p=sample_prob)
        batch_xs = self.xs[chosen_inds]
        batch_ys = self.ys[chosen_inds]
        return batch_xs, batch_ys






def rot_and_shift(img, angle_max, x_shift_max, y_shift_max):
    angle = tf.random.uniform([1], minval=-angle_max, maxval=angle_max)
    x_shift = tf.random.uniform([1], minval=-x_shift_max, maxval=x_shift_max)
    y_shift = tf.random.uniform([1], minval=-y_shift_max, maxval=y_shift_max)

    img = tf.contrib.image.rotate(
        img,
        angle * np.pi / 180,
        interpolation='BILINEAR'
    )
    # random shift
    transforms = tf.concat([tf.constant(1., shape=[1]), tf.constant(0., shape=[1]), -x_shift, tf.constant(0., shape=[1]), tf.constant(1., shape=[1]), -y_shift, tf.constant(0., shape=[1]), tf.constant(0., shape=[1])], 0)
    img = tf.contrib.image.transform(img, transforms, interpolation='NEAREST')
    return img

def rot_and_shift_exact(img, angle, x_shift, y_shift):
    img = tf.contrib.image.rotate(
        img,
        angle * np.pi / 180,
        interpolation='BILINEAR'
    )
    # random shift
    transforms = [1, 0, -x_shift, 0, 1, -y_shift, 0, 0]
    img = tf.contrib.image.transform(img, transforms, interpolation='NEAREST')
    return img


# TBD: implement transformations
def transform(images, mode, angle, x_shift, y_shift, flip=False, standardize=True):


    if mode == 'augmentation':  # training
        # Randomly crop a [height, width] section of the image.
        #images = tf.random_crop(images, [IMAGE_SIZE, IMAGE_SIZE, channel_num])
        # Randomly flip the image horizontally.
        images = tf.image.random_flip_left_right(images)
        angle = 0
        x_shift = 2
        y_shift = 2
        images = rot_and_shift(images, angle, x_shift, y_shift)
        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        images = tf.image.random_brightness(images, max_delta=63)
        images = tf.image.random_contrast(images, lower=0.2, upper=1.8)

    elif mode == 'plain':  # Image processing for evaluation.
        # Crop the central [height, width] of the image. Currently, this is doing nothing
        #images = tf.image.resize_image_with_crop_or_pad(images, IMAGE_SIZE, IMAGE_SIZE)
        pass
    elif mode == 'rot_and_shift':
        # Crop the central [height, width] of the image.
        #images = tf.image.resize_image_with_crop_or_pad(images, IMAGE_SIZE, IMAGE_SIZE)
        # Additional preprocessing is needed when training
        if flip:
            images = tf.image.random_flip_left_right(images)
        images = rot_and_shift(images, angle, x_shift, y_shift)

    elif mode == 'rot_and_shift_exact':
        # Crop the central [height, width] of the image.
        #images = tf.image.resize_image_with_crop_or_pad(images, IMAGE_SIZE, IMAGE_SIZE)

        images = rot_and_shift_exact(images, angle, x_shift, y_shift)

    else:
        raise

    if standardize:
        images = tf.image.per_image_standardization(images)

    return images


if __name__ == '__main__':
    import json
    with open('configs/config_svhn.json') as config_file:
        config = json.load(config_file)

    data_path = config['data_path']
    dataload = SVHNData(data_path)

    batch_size = 5

    with tf.Session(config=tf.ConfigProto()) as sess:

        for i in range(5):
            x_batch, y_batch = dataload.eval_data.get_next_batch(batch_size,
                                                                   multiple_passes=True)

            print(y_batch[0])
            plt.imshow(x_batch[0]/255)
            plt.show()
