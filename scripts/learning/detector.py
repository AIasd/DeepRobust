import tensorflow as tf

class ModelDetector(object):
    def __init__(self, task_type='regression', bin_number=10):
        '''
        task_type: ['regression', 'classification']
        bin_number is valid only when task_type == classification
        '''
        assert task_type in ['regression', 'classification']
        self.precision = tf.float32
        self.label_smoothing = 0
        self.task_type = task_type
        self.bin_number = bin_number

    def _encoder(self, x_input, y_in, is_train):
        """Build the core model within the graph."""
        with tf.variable_scope('detector_encoder', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('input'):
                  self.x_input = x_input
                  self.y_input = y_in
                  self.is_training = is_train

            with tf.variable_scope('1st'):
                x = self._fully_connected(self.x_input, 1500)
                x = self._relu(x, 0.1)

            with tf.variable_scope('2nd'):
                x = self._fully_connected(x, 1000)
                x = self._relu(x, 0.1)

            with tf.variable_scope('3rd'):
                x = self._fully_connected(x, 500)
                x = self._relu(x, 0.1)

            with tf.variable_scope('logit'):
                pre_softmax = None
                if self.task_type == 'regression':
                    pre_softmax = self._fully_connected(x, 1)
                    # pre_softmax = tf.clip_by_value(pre_softmax, 0, 1)
                elif self.task_type == 'classification':
                    pre_softmax = self._fully_connected(x, self.bin_number)

        mean_loss, weight_decay_loss, mean_error = None, None, None
        if self.task_type == 'regression':
            self.y_input = tf.cast(self.y_input, tf.float32)
            mean_error = tf.reduce_mean(tf.abs(pre_softmax - self.y_input))
            with tf.variable_scope('detector_costs'):
                mean_loss = tf.losses.mean_squared_error(self.y_input, pre_softmax)
                weight_decay_loss = self._decay()
            predictions = pre_softmax
        elif self.task_type == 'classification':
            ce_labels = tf.one_hot(self.y_input, self.bin_number)
            y_xent = tf.losses.softmax_cross_entropy(
                onehot_labels=ce_labels,
                logits=pre_softmax,
                label_smoothing=self.label_smoothing)
            predictions = tf.argmax(pre_softmax, 1)
            correct_prediction = tf.equal(predictions, self.y_input)
            mean_error = 1 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            with tf.variable_scope('detector_costs'):
                mean_loss = tf.reduce_mean(y_xent)
                weight_decay_loss = self._decay()

        layer_values = {}

        return [layer_values, mean_loss, weight_decay_loss, mean_error, predictions]

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
          if var.op.name.find('DW') > 0:
            costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer"""
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
