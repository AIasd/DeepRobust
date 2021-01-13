# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf



BN_EPSILON = 0.001
weight_decay = 0.0002


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TBD: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def fc_layer(input_layer, num_output, is_relu=True):
    '''
    full connection layer
    :param input_layer: 2D tensor
    :param num_output: number of output layer
    :param is_relu: judge use activation function: relu
    :return: output layer, 2D tensor
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_output], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_output], initializer=tf.zeros_initializer())

    fc_result = tf.matmul(input_layer, fc_w) + fc_b
    if is_relu is True:
        return tf.nn.relu(fc_result)
    else:
        return fc_result


def fc_bn_layer(input_layer, num_output, is_relu=True):
    '''
    full connection layer
    :param input_layer: 2D tensor
    :param num_output: number of output layer
    :param is_relu: judge use activation function: relu
    :return: output layer, 2D tensor
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_output], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_output], initializer=tf.zeros_initializer())

    fc_result = tf.matmul(input_layer, fc_w) + fc_b
    fc_bn_layer = batch_fc_normalization_layer(fc_result, num_output)
    if is_relu is True:
        return tf.nn.relu(fc_bn_layer)
    else:
        return fc_bn_layer





def batch_fc_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation of full connection layer
    :param input_layer: 2D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 2D tensor
    :return: the 2D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    fc_bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return fc_bn_layer


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def conv_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(conv(X))
    '''
    filter = create_variables(name='conv_relu', shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    output = tf.nn.relu(conv_layer)

    return output


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv_bn_relu', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='bn_relu_conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



class ModelVani(object):
    """ResNet model."""
    def __init__(self, dataset, retrain_with_weak_points=False):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.retrain_with_weak_points = retrain_with_weak_points
        self.precision = tf.float32
        self.ratio = 0.1
        self.dataset = dataset

        self.output_classes = 10
        self.in_filter_size = 3
        if self.dataset in ['cifar100']:
            self.output_classes = 100
        elif self.dataset in ['celeb5']:
            self.output_classes = 5
        elif self.dataset in ['celebA']:
            self.output_classes = 100

        if self.dataset in ['fmnist']:
            self.in_filter_size = 1


    def _encoder(self, x_input, y_in, is_train, sample_prob_slice):
        with tf.variable_scope('main_encoder', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('input'):

                self.x_input = x_input
                self.y_input = y_in
                self.is_training = is_train
                self.sample_prob_slice = sample_prob_slice
            # block1
            with tf.variable_scope('conv1_1'):
                conv1_1 = conv_bn_relu_layer(self.x_input, [3, 3, self.in_filter_size, 64], 1)
            with tf.variable_scope('conv1_2'):
                conv1_2 = conv_bn_relu_layer(conv1_1, [3, 3, 64, 64], 1)
            with tf.name_scope('conv1_max_pool'):
                conv2 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # block2
            with tf.variable_scope('conv2_1'):
                conv2_1 = conv_bn_relu_layer(conv2, [3, 3, 64, 128], 1)
            with tf.variable_scope('conv2_2'):
                conv2_2 = conv_bn_relu_layer(conv2_1, [3, 3, 128, 128], 1)
            with tf.name_scope('conv2_max_pool'):
                conv3 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # block3
            with tf.variable_scope('conv3_1'):
                conv3_1 = conv_bn_relu_layer(conv3, [3, 3, 128, 256], 1)
            with tf.variable_scope('conv3_2'):
                conv3_2 = conv_bn_relu_layer(conv3_1, [3, 3, 256, 256], 1)
            with tf.variable_scope('conv3_3'):
                conv3_3 = conv_bn_relu_layer(conv3_2, [3, 3, 256, 256], 1)
            # with tf.variable_scope('conv3_4'):
            #     conv3_4 = conv_bn_relu_layer(conv3_3, [3, 3, 256, 256], 1)
            with tf.name_scope('conv3_max_pool'):
                conv4 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # block4
            with tf.variable_scope('conv4_1'):
                conv4_1 = conv_bn_relu_layer(conv4, [3, 3, 256, 512], 1)
            with tf.variable_scope('conv4_2'):
                conv4_2 = conv_bn_relu_layer(conv4_1, [3, 3, 512, 512], 1)
            with tf.variable_scope('conv4_3'):
                conv4_3 = conv_bn_relu_layer(conv4_2, [3, 3, 512, 512], 1)
            # with tf.variable_scope('conv4_4'):
            #     conv4_4 = conv_bn_relu_layer(conv4_3, [3, 3, 512, 512], 1)
            with tf.name_scope('conv4_max_pool'):
                conv5 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # block5
            with tf.variable_scope('conv5_1'):
                conv5_1 = conv_bn_relu_layer(conv5, [3, 3, 512, 512], 1)
            with tf.variable_scope('conv5_2'):
                conv5_2 = conv_bn_relu_layer(conv5_1, [3, 3, 512, 512], 1)
            with tf.variable_scope('conv5_3'):
                conv5_3 = conv_bn_relu_layer(conv5_2, [3, 3, 512, 512], 1)
            # with tf.variable_scope('conv5_4'):
            #     conv5_4 = conv_bn_relu_layer(conv5_3, [3, 3, 512, 512], 1)
            with tf.name_scope('conv5_max_pool'):
                conv6 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # full connection layer
            # fc_shape = conv6.get_shape().as_list()
            s = tf.shape(conv6)[0]
            # nodes = fc_shape[1]*fc_shape[2]*fc_shape[3]
            # nodes = tf.shape(conv6)[1]*tf.shape(conv6)[2]*tf.shape(conv6)[3]

            # 512 is hardcoded for CIFAR-10, SVHN, and FMNIST
            fc_reshape = tf.reshape(conv6, (s, 512), name='fc_reshape')
            # x3 = fc_reshape

            # fc6
            with tf.variable_scope('fc6'):
                fc6 = fc_bn_layer(fc_reshape, 4096)
            with tf.name_scope('dropout1'):
                fc6_drop = tf.nn.dropout(fc6, 0.5)
            # fc7
            with tf.variable_scope('fc7'):
                fc7 = fc_bn_layer(fc6_drop, 4096)
            with tf.name_scope('dropout2'):
                fc7_drop = tf.nn.dropout(fc7, 0.5)

            x4 = fc7_drop
            # fc8
            with tf.variable_scope('fc8'):
                fc8 = fc_bn_layer(fc7_drop, self.output_classes, is_relu=False)

            pre_softmax = fc8

            predictions = tf.argmax(pre_softmax, 1)
            correct_prediction = tf.equal(predictions, self.y_input)
            mask = tf.cast(correct_prediction, tf.int64)
            num_correct = tf.reduce_sum(
                tf.cast(correct_prediction, tf.int64))
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

            def f1(): return self.y_xent * self.sample_prob_slice
            def f2(): return self.y_xent

            with tf.variable_scope('costs'):
              self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=pre_softmax, labels=self.y_input)
              if self.retrain_with_weak_points:
                  self.y_xent = tf.cond(self.is_training, f1, f2)
              mean_xent = tf.reduce_mean(self.y_xent)
              weight_decay_loss = self._decay()

            layer_values = {'x0':None, 'x1':None, 'x2':None, 'x3':None, 'x4':x4, 'pre_softmax':pre_softmax, 'softmax': tf.nn.softmax(pre_softmax)}

            return [layer_values, mean_xent, weight_decay_loss, num_correct, accuracy, predictions, mask]

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
          if var.op.name.find('fc_weights') >= 0 or var.op.name.find('conv_bn_relu') >= 0:
            costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)



# def test_graph(train_dir='logs'):
#     '''
#     Run this function to look at the graph structure on tensorboard. A fast way!
#     :param train_dir:
#     '''
#     input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
#     result = inference_VGG(input_tensor, reuse=False)
#     init = tf.initialize_all_variables()
#     sess = tf.Session()
#     sess.run(init)
#     summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
