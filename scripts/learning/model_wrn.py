import numpy as np
import tensorflow as tf

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

    self.label_smoothing = 0
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

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _encoder(self, x_input, y_in, is_train, sample_prob_slice):
    """Build the core model within the graph."""


    ##TODO: need to be changed according to mnist model: the mask, and the bn

    with tf.variable_scope('main_encoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('input'):

              self.x_input = x_input
              self.y_input = y_in
              self.is_training = is_train
              self.sample_prob_slice = sample_prob_slice

              input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                       self.x_input)
              x0 = self._conv('init_conv', input_standardized, 3, self.in_filter_size, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        self.filters = [16, 32, 64, 128]
        filters = self.filters

        # Update hps.num_residual_units to 9
        with tf.variable_scope('unit_1_0'):
          x = res_func(x0, filters[0], filters[1], self._stride_arr(strides[0]),
                       activate_before_residual[0])
        for i in range(1, 5):
          with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        x1 = x
        with tf.variable_scope('unit_2_0'):
          x = res_func(x1, filters[1], filters[2], self._stride_arr(strides[1]),
                       activate_before_residual[1])
        for i in range(1, 5):
          with tf.variable_scope('unit_2_%d' % i):
            x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        x2 = x
        with tf.variable_scope('unit_3_0'):
          x = res_func(x2, filters[2], filters[3], self._stride_arr(strides[2]),
                       activate_before_residual[2])
        for i in range(1, 5):
          with tf.variable_scope('unit_3_%d' % i):
            x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        x3 = x
        # Extract information from mid layer
        x3_extracted = tf.reduce_mean(x3, axis=[1, 2])

        with tf.variable_scope('unit_last'):
          x = self._batch_norm('final_bn', x3)
          x = self._relu(x, 0.1)
          x = self._global_avg_pool(x)

        x4 = x
        with tf.variable_scope('logit'):
          pre_softmax = self._fully_connected_final(x, self.output_classes)

    predictions = tf.argmax(pre_softmax, 1)
    correct_prediction = tf.equal(predictions, self.y_input)
    mask = tf.cast(correct_prediction, tf.int64)
    num_correct = tf.reduce_sum(mask)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def f1(): return self.y_xent * self.sample_prob_slice
    def f2(): return self.y_xent

    with tf.variable_scope('costs'):
        ce_labels = tf.one_hot(self.y_input, self.output_classes)
        self.y_xent = tf.losses.softmax_cross_entropy(
            onehot_labels=ce_labels,
            logits=pre_softmax,
            label_smoothing=self.label_smoothing)
        if self.retrain_with_weak_points:
            self.y_xent = tf.cond(self.is_training, f1, f2)
        mean_xent = tf.reduce_mean(self.y_xent)
        weight_decay_loss = self._decay()

    layer_values = {'x0':x0, 'x1':x1, 'x2':x2, 'x3':x3_extracted, 'x4':x4, 'pre_softmax':pre_softmax, 'softmax': tf.nn.softmax(pre_softmax)}

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
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          self.precision, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n), dtype=self.precision))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim, name, in_dim=-1):
    """FullyConnected layer for final output."""
    with tf.variable_scope(name):
        prod_non_batch_dimensions=1
        if in_dim == -1:
            num_non_batch_dimensions = len(x.shape)
            prod_non_batch_dimensions = 1
            for ii in range(num_non_batch_dimensions - 1):
              prod_non_batch_dimensions *= int(x.shape[ii + 1])

        else:
            prod_non_batch_dimensions = in_dim
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim], dtype=self.precision,
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0, dtype=self.precision))
        b = tf.get_variable('biases', [out_dim], dtype=self.precision,
                            initializer=tf.constant_initializer(dtype=self.precision))
        return tf.nn.xw_plus_b(x, w, b)


  def _fully_connected_final(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.initializers.variance_scaling(distribution='uniform', dtype=self.precision))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer(dtype=self.precision))
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
