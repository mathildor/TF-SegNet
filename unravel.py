import os # provides a portable way of using operating system dependent functionality, example read file
import re
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import numpy as np
import time
import math
from math import ceil
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import dtypes

""" Test script for unravel and upsampling with max pool indices """

init_op = tf.global_variables_initializer()
out_shape = [2,4,4,4]
image_pl = tf.placeholder(tf.float32, shape=(2, 4, 4, 4))

pool1, indices = tf.nn.max_pool_with_argmax(image_pl, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME', name='pool1')

indices_flat_batches = tf.reshape(indices, [out_shape[0],-1]) #flatten each batch independently

""" Unravel indices - define the operations"""
tot_num_indices_per_batch = out_shape[1] * out_shape[2] * out_shape[3]
indices_per_batch = tf.to_int32(tf.divide(tot_num_indices_per_batch, 4))
# unraveled_indices = tf.placeholder(tf.int64, shape=(2,4))
# unraveled_indices=[]
for i in range(0, out_shape[0]): #for each batch
  print(i)
  indices = indices_flat_batches[i]
  batch_dim = tf.multiply(tf.ones([indices_per_batch], tf.int64), i) #index starts with zero for every new batch
  #Finding index as if always in batch 1 -> will still have same last three dimensions
  first_dim = tf.to_int64(tf.divide(indices, (out_shape[2] * out_shape[3])))
  #finding index as if in "first" matrix - will still have same two last dimensions
  first_matr_indices = indices - (first_dim * out_shape[2] * out_shape[3])
  second_dim = tf.to_int64(tf.divide(first_matr_indices, out_shape[3]))
  third_dim = tf.subtract(first_matr_indices, (out_shape[3] * (first_matr_indices // out_shape[3]) ))
  res_index = tf.transpose([batch_dim, first_dim, second_dim, third_dim])
  if(i>0):
    print("\n\n ''''''' i is more than zero")
    unraveled_indices = tf.concat([unraveled_indices, res_index], 0)
  else:
    print("\n\n ''''''' init res")
    unraveled_indices = res_index
values_flattened = tf.reshape(pool1, [-1])
result_matrix = tf.SparseTensor(tf.to_int64(unraveled_indices), tf.to_int64(values_flattened), tf.to_int64(out_shape))
res_matrix_dense = tf.sparse_tensor_to_dense(result_matrix, name="sparse_tensor", validate_indices=False)

""" Run the method """
with tf.Session() as s:
  s.run(init_op)

  image = np.random.random_integers(0,10,(2,4,4,4)) #Batch, height, width, channels/depth
  print('original randomly created matrix')
  print(image)

  flattened_values = tf.reshape(image, [-1])

  print("Indices calculated from argmax:")
  print(s.run(indices, feed_dict={image_pl: image}))
  print("batch dim:")
  print(s.run(batch_dim, feed_dict={image_pl: image}))
  print("first dim:")
  print(s.run(first_dim, feed_dict={image_pl: image}))
  print("second dim:")
  print(s.run(second_dim, feed_dict={image_pl: image}))
  print("third dim:")
  print(s.run(third_dim, feed_dict={image_pl: image}))
  print("RESULT INDEX:")
  print(s.run(unraveled_indices, feed_dict={image_pl: image}))
  print("FLATTENED:")
  print(s.run(indices_flat_batches, feed_dict={image_pl: image}))
  argmax_indices = s.run(indices, feed_dict={image_pl: image})

  print("unravel indices result")
  # print(s.run(tf.transpose(unraveled_indices), feed_dict={image_pl:image}))
  print("NUMPY unravel result")
  image = np.array(image)
  unraveled_indices_numpy = np.unravel_index(argmax_indices, image.shape)
  print(unraveled_indices_numpy)
  #
  print("res matrix:")
  # print(s.run(result_matrix, feed_dict={image_pl: image}))
  print(s.run(res_matrix_dense, feed_dict={image_pl: image}))
