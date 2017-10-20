import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret


def conv_classifier(input_layer, initializer):
    # output predicted class number (2)
    with tf.variable_scope('conv_classifier') as scope: #all variables prefixed with "conv_classifier/"
        shape=[1, 1, 64, FLAGS.num_class]
        kernel = _variable_with_weight_decay('weights', shape=shape, initializer=initializer, wd=None)
        #kernel = tf.get_variable('weights', shape, initializer=initializer)
        conv = tf.nn.conv2d(input_layer, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FLAGS.num_class], tf.constant_initializer(0.0))
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
    return conv_classifier


def conv_layer_with_bn(initializer, inputT, shape, is_training, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]

    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=shape, initializer=initializer, wd=None)
        #kernel = tf.get_variable(scope.name, shape, initializer=initializer)
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[out_channel], dtype=tf.float32),
                       trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)

        if activation is True: #only use relu during encoder
            conv_out = tf.nn.relu(batch_norm_layer(bias, is_training, scope.name))
        else:
            conv_out = batch_norm_layer(bias, is_training, scope.name)
    return conv_out

def batch_norm_layer(inputT, is_training, scope):
      return tf.cond(is_training,
            lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, decay=FLAGS.moving_average_decay, scope=scope),
            lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           center=False, reuse = True, decay=FLAGS.moving_average_decay, scope=scope))



def _variable_with_weight_decay(name, shape, initializer, wd):
    """ Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(name, shape, initializer)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32 #added this after, cause it was in cifar model
        var = tf.get_variable(name, shape, initializer=initializer)#, dtype=dtype)
    return var