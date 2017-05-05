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

def unravel_index(indices, shape):
  with tf.name_scope('unravel_index'):
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    strides = tf.cumprod(shape, reverse=True)
    strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
    return (indices // strides_shifted) % strides

    #Example:
        #out = unravel_index([22, 41, 37], (7, 6))
        #print(s.run(out))
            # ==> [[3 6 6]
            #   [4 5 1]]

def upsample(values, indices, out_shape):
  #flatten values into one dimension
  # indices = tf.reshape(indices, [-1])

  unraveled_indices = tf.transpose(unravel_index(indices, out_shape)) #Tranpose result, because it directly gives correct indices for result matrix
  print("unraveled indices")
  # print(s.run(tf.transpose(unraveled_indices)))
  values_flattened = tf.reshape(values, [-1])
  print('values_flattened')
  # print(s.run(values_flattened))

  result_matrix = tf.SparseTensor(tf.to_int64(unraveled_indices), tf.to_int64(values_flattened), tf.to_int64(out_shape))
  res_matrix_dense = tf.sparse_tensor_to_dense(result_matrix, name="sparse_tensor", validate_indices=False)

  print("RESULT:")
  print(res_matrix_dense)
  # print(s.run(res_matrix_dense))
  # print(res_matrix_dense.get_shape)
  # res_matrix_dense.eval()
  # print(s.run(matrix_shape))

  return res_matrix_dense


init_op = tf.global_variables_initializer()

#read image:
# image_filename = "../aerial_datasets/IR_RGB_0.1res/RGB_images/combined_dataset/test_images/images/map_color_253.png"
# imageValue = tf.read_file(image_filename)
# image_bytes = tf.image.decode_png(imageValue)
# image = tf.reshape(image_bytes, (512, 512, 3))

x = tf.placeholder(tf.float32, shape=(1, None, None, None))

pool1, indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME', name='pool1')
out_shape = tf.to_int64([4, 2, 4])
# unpool = upsample(tf.to_int64(pool1), indices, size)

indices_flat = tf.reshape(indices, [-1])

unraveled_indices = tf.transpose(unravel_index(indices_flat, out_shape)) #Tranpose result, because it directly gives correct indices for result matrix
values_flattened = tf.reshape(pool1, [-1])
# print(s.run(values_flattened))

result_matrix = tf.SparseTensor(tf.to_int64(unraveled_indices), tf.to_int64(values_flattened), tf.to_int64(out_shape))
res_matrix_dense = tf.sparse_tensor_to_dense(result_matrix, name="sparse_tensor", validate_indices=False)

## TESTING
image =np.array([[[1, 2, 1, 5], [7, 1, 5, 8]],
        [[9, 2, 3, 5], [5, 1, 6, 7]],
        [[8, 4, 3, 5], [1, 1, 6, 8]],
        [[3, 2, 9, 5], [5, 3, 4, 1]]])

indices_calced = [[
                  [[8, 1, 14, 7]],
                  [[16, 17, 26, 23]]
                 ]]
# indices_calced = [8, 1, 14, 17, 16, 17, 26, 23]

print("unravel answer numpy:")
print(np.unravel_index(indices_calced, image.shape))


out_shape= [1,4,2,4] #height, widt, depth
with tf.Session() as s:
  s.run(init_op)
  # image = np.float32(cv2.imread(image_filename))
  image =[[[1, 2, 1, 5], [7, 1, 5, 8]],
          [[9, 2, 3, 5], [5, 1, 6, 7]],
          [[8, 4, 3, 5], [1, 1, 6, 8]],
          [[3, 2, 9, 5], [5, 3, 4, 1]]]
  print(s.run(tf.reshape(image, [-1])))
  flattened_values = tf.reshape(image, [-1])

  pool1.eval(feed_dict={x: [image]})
  print("Indices calculated from argmax:")
  print(s.run([indices], feed_dict={x: [image]}))
  print("pooling values - result matrix after pooling:")
  print(pool1.eval(feed_dict={x: [image]}))
  # print("Upsampeled matrix ")
  # print(s.run([unpool], feed_dict={x: [image]}))
  print("Unravel indices:")
  print(s.run([unraveled_indices], feed_dict={x: [image]}))
  # print(s.run([unpool], feed_dict={x: [image]}))

  # unr = unravel_index(indices, shape)
  # #print(s.run(unr))
  # print("true answer:")
  # print(np.unravel_index(indices, shape))

  # upsample(matrix_to_upsample, indices, shape)






  #
