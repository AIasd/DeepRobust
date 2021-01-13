# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class ModelVani(object):
  """ResNet model."""

  def __init__(self, dataset, retrain_with_weak_points=False):
    """ResNet constructor.
    """
    self.retrain_with_weak_points = retrain_with_weak_points
    self.filters = [16, 16, 32, 64]
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


  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _encoder(self, x_input, y_in, is_train, sample_prob_slice):
    """Build the core model within the graph."""
    filters = self.filters


    with tf.variable_scope('main_encoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('input'):

          self.x_input = x_input
          self.y_input = y_in
          self.is_training = is_train
          self.sample_prob_slice = sample_prob_slice

          x = self._conv('init_conv', self.x_input, 3, self.in_filter_size, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        with tf.variable_scope('unit_1_0'):
          x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                       activate_before_residual[0])
        for i in range(1, 5):
          with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
          x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                       activate_before_residual[1])
        for i in range(1, 5):
          with tf.variable_scope('unit_2_%d' % i):
            x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
        x2 = x
        with tf.variable_scope('unit_3_0'):
          x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                       activate_before_residual[2])
        for i in range(1, 5):
          with tf.variable_scope('unit_3_%d' % i):
            x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
        x3 = x
        # Extract information from mid layer
        x3_extracted = tf.reduce_mean(x3, axis=[1, 2])

        with tf.variable_scope('unit_last'):
          x = self._batch_norm('final_bn', x)
          x = self._relu(x, 0.1)
          x = self._global_avg_pool(x)
        x4 = x

        # uncomment to add and extra fc layer
        # with tf.variable_scope('unit_fc'):
        #   x = self._fully_connected(x, 1024)
        #   x = self._relu(x, 0.1)
        with tf.variable_scope('logit'):
          pre_softmax = self._fully_connected(x, self.output_classes)

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

    layer_values = {'x0':None, 'x1':None, 'x2':x2, 'x3':x3_extracted, 'x4':x4, 'pre_softmax':pre_softmax, 'softmax': tf.nn.softmax(pre_softmax)}

    return [layer_values, mean_xent, weight_decay_loss, num_correct, accuracy, predictions, mask]

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=self.is_training)

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') >= 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
