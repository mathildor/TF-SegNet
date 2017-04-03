
"""
Builds the model

-- Summary of functions --

Building the graph:
  inference()
  loss()
  train()

"""


""" Importing libraries """

import os # provides a portable way of using operating system dependent functionality, example read file
import re
import sys

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import numpy as np
import math
from datetime import datetime
import time

from PIL import Image

#from tensorflow.python import control_flow_ops

from math import ceil
from tensorflow.python.ops import gen_nn_ops
import skimage
import skimage.io

FLAGS = tf.app.flags.FLAGS

# modules
import Utils
from Inputs import *



#creating own gradient function that does not exist default
  # not sure why and where it is used? Somewhere in the deconvolution?
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')

""" Found this here: https://github.com/tensorflow/tensorflow/issues/1793. maybe this is better?
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
  return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                               grad,
                                               op.outputs[1],
                                               op.get_attr("ksize"),
                                               op.get_attr("strides"),
                                               padding=op.get_attr("padding"))
"""


# Constants describing the training process.
#NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
#LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

#EVAL_BATCH_SIZE = 1
#BATCH_SIZE = 1
#READ_DATA_SIZE = 100

# for CamVid
#IMAGE_HEIGHT = 360
#IMAGE_WEIGHT = 480
#IMAGE_DEPTH = 3 #channels, here RGB

#NUM_CLASSES = 11

"""
# Let's load a previously saved meta graph in the default graph
# This function returns a Saver
saver = tf.train.import_meta_graph('results/model.ckpt-1000.meta')

# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()

# Finally we can retrieve tensors, operations, collections, etc.
global_step_tensor = graph.get_tensor_by_name('loss/global_step:0')
train_op = graph.get_operation_by_name('loss/train_op')
hyperparameters = tf.get_collection('hyperparameters')

"""


def inference(images, phase_train, batch_size):
  """ Inference = slutning. It builds the graph as far as is required for running the network forward
      to make predictions.

      Builds the model. The arcitecure has 4 sets of sizes on layers, each appears twice
      - once in the encoder and once in the decoder. Each "block" of layers (with the sames size)
      are of different types. Block one for example has two conv-batch-relu layer and one pooling layer.

      Args:
        images: Images Tensors
        phase_train:

      Returns:
        logit (scores for the classes, that sums up to 1)
  """
  # norm1
    #tf.nn.lrn = local response normalisation
  norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                         name='norm1')
  # conv1
    #input to: (inputT, shape, train_phase, activation=True, name=None)
    #shape is used to create kernel (kernel is the filter that will be convolved over the input)
    #shape = [patch_size_width, patch_size_heigh, input_channels, output_channels]
    #input_channels are three since the images has three channels => IMAGE_DEPTH=3
  conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
  # pool1
    #max_pool_with_argmax: Args: input tensor to pool over, ksize=window size for input tensor.
    #strides = [bach_size, image_rows, image_cols, number_of_colors].
    #[1,2,2,1] -> want to apply the filters on every second row and column.
  pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool1')
  # conv2
    #Since output_channels of conv1 was 64, input_channels for conv2 is 64
  conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")

  # pool2
  pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  # conv3
  conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")

  # pool3
  pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')
  # conv4
  conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

  # pool4
  pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool4')
  """ End of encoder """

  """ start upsample """
  # upsample4
  # Need to change when using different dataset out_w, out_h
  # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
  upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4")
  # decode 4
  conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

  # upsample 3
  # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
  upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 90, 120, 64], 2, "up3")
  # decode 3
  conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

  # upsample2
  # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
  upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 180, 240, 64], 2, "up2")
  # decode 2
  conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

  # upsample1
  # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
  upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
  # decode4
  conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
  """ end of Decode """

  """ Start Classify """
  # output predicted class number (6)
  with tf.variable_scope('conv_classifier') as scope: #all variables prefixed with "conv_classifier/"
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 64, FLAGS.num_class],
                                         initializer=msra_initializer(1, 64),
                                         wd=0.0005)
    conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [FLAGS.num_class], tf.constant_initializer(0.0))
    conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name) #tf.nn.bias_add is an activation function. Simple add that specifies 1-D tensor bias
  #logit = conv_classifier

  return conv_classifier


def loss(logits, labels):
  """
  !! Not used for now, but might use if dataset used is not unbalanced !!

  Adds to the inference graph the ops required to generate loss.
  Here the one-hot-encoding is done.

      loss func without re-weighting
  """
  # Calculate the average cross entropy loss across the batch.
  logits = tf.reshape(logits, (-1, FLAGS.num_class))
  labels = tf.reshape(labels, [-1])

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( #one-hot-encoding
      labels=labels, logits=logits, name='cross_entropy_per_example')
  #reduce mean -> average the cross entropy accross the batch dimension
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def weighted_loss(logits, labels, num_classes, head=None): #None is default value (if no other is given)
  """Calculate the loss from the logits and the labels.
  Args:
    logits: tensor, float - [batch_size, width, height, num_classes].
        Use vgg_fcn.up as logits.
    labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
        The ground truth of your data.
    head: numpy array - [num_classes]
        Weighting the loss of each class
        Optional: Prioritize some classes
  Returns:
    loss: Loss tensor of type float.
  """
  with tf.name_scope('loss'):

      logits = tf.reshape(logits, (-1, num_classes))

      epsilon = tf.constant(value=1e-10)

      logits = logits + epsilon

      # construct one-hot label array
      label_flat = tf.reshape(labels, (-1, 1))

      # should be [batch ,num_classes]
      labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

      softmax = tf.nn.softmax(logits)

      cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), reduction_indices=[1])

      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

      tf.add_to_collection('losses', cross_entropy_mean)

      loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  return loss

def cal_loss(logits, labels):
  """ Assigning loss_weight and using weighted loss because of unbalanced dataset """
  loss_weight = np.array([
    0.2595,
    0.1826,
    4.5640,
    0.1417,
    0.9051,
    0.3826,
    9.6446,
    1.8418,
    0.6823,
    6.2478,
    7.3614,
  ]) # class 0~10

  labels = tf.cast(labels, tf.int32)

  return weighted_loss(logits, labels, num_classes=FLAGS.num_class, head=loss_weight)

def train(total_loss, global_step):

  """ Training the SegNet model
    This train method is only for building the graph - defines the part of the graph
    that is needed for training. The actual training is done in the train() in model_train.py.

    ? Adds to the loss graph the ops required to compute and apply gradients.?

    Create an optimizer and apply to all trainable variables.
    Add moving average for all trainable variables??

    Args:
      total_loss: Total loss from loss(), -  or cal_loss() in this case?
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.

  """

  #total_sample = 274

  #Variables that affect learning rate.
  num_batches_per_epoch = 274/1

  """ fixed learning rate """
  lr = FLAGS.learning_rate #Setting constant learning rate as it is most common with adam optimizer.

  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  #Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      FLAGS.moving_average_decay, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


""" --- Initializers --- """

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    Why called msra????

    Truncated normal distribution
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


""" ----  Summaries  -----"""

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _add_loss_summaries(total_loss):
  """Add summaries for losses.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


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

def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.
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
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
  """
  Used in inference() to define conv-layers with batch normalisation and ReLu (blue box in figure).
  """


  in_channel = shape[2]
  out_channel = shape[3]
  k_size = shape[0]
  with tf.variable_scope(name) as scope:

    kernel = _variable_with_weight_decay('weights',
                                         shape=shape,
                                         initializer=msra_initializer(k_size, in_channel),
                                         wd=None)

    #kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
    conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    if activation is True:
      conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
    else:
      conv_out = batch_norm_layer(bias, train_phase, scope.name)
  return conv_out

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  """Used in inference() to create upsample layer"""
  # output_shape = [batch, width, height, channels]
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  """Used in conv_layer_with_bn()"""
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn - hvorfor?

    Used by the deconv_layer() to define the weights?

    Args:
      f_shape:? example is [2, 2, 64, 64], but not sure what it defines

    Returns:
      variable: named up_filter
  """
  width = f_shape[0]
  height = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(height):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)


def dump_unravel(indices, shape):
    """
    self-implented unravel indice, missing gradients, need fix
    """
    N = indices.get_shape().as_list()[0]
    tb = tf.constant([shape[0]], shape=[1,N])
    ty = tf.constant([shape[1]], shape=[1,N])
    tx = tf.constant([shape[2]], shape=[1,N])
    tc = tf.constant([shape[3]], shape=[1,N])

    c = indices % tc
    x = ((indices - c) // tc ) % tx
    t_temp = ((indices - c) // tc)
    y = ((t_temp - x) // tx) % ty
    t_temp = ((t_temp - x) // tx)
    b = (t_temp - y) // ty

    t_new = tf.transpose(tf.reshape(tf.pack([b,y,x,c]), (4, N)))
    return t_new

def upsample_with_pool_indices(value, indices, shape=None, scale=2, out_w=None, out_h=None,name="up"):
    s = shape.as_list()
    b = s[0]
    w = s[1]
    h = s[2]
    c = s[3]
    if out_w is not None:
      unraveled = dump_unravel(tf.to_int32(tf.reshape(indices,[b*w*h*c])), [b, out_w, out_h, c])
      ts = tf.SparseTensor(indices=tf.to_int64(unraveled), values=tf.reshape(value, [b*w*h*c]), shape=[b,out_w,out_h,c])
    else:
      unraveled = dump_unravel(tf.to_int32(tf.reshape(indices,[b*w*h*c])), [b, w*scale, h*scale, c])
      ts = tf.SparseTensor(indices=tf.to_int64(unraveled), values=tf.reshape(value, [b*w*h*c]), shape=[b,w*scale,h*scale,c])

    t_dense = tf.sparse_tensor_to_dense(ts, name=name, validate_indices=False)
    return t_dense
