import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
import glob

from PIL import Image #python 3


import skimage
import skimage.io
import SimpleITK as sitk

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3

BATCH_SIZE = 1

NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1



def run():
  print('running run function')
  batch_size = 1
  train_dir = "./tmp/logs"
  #image_filenames, label_filenames = get_filename_list("tmp3/first350/SegNet-Tutorial/CamVid/train.txt")
  image_filenames, label_filenames = get_filename_list("./dataset/dummy_set/train/images")
  #val_image_filenames, val_label_filenames = get_filename_list("tmp3/first350/SegNet-Tutorial/CamVid/val.txt")
  val_image_filenames, val_label_filenames = get_filename_list("./dataset/dummy_set//val/images")

  with tf.Graph().as_default():

    train_data_node = tf.placeholder(
          tf.float32,
          shape=[batch_size, 360, 480, 3])

    train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, 360, 480, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    global_step = tf.Variable(0, trainable=False)

    # For CamVid
      #Make images into correct type(float32/float16 el.), create shuffeled batches ++
    images, labels = datasetInputs(image_filenames, label_filenames, BATCH_SIZE) #prev name for datasetInputs = CamVidInputs

    val_images, val_labels = datasetInputs(val_image_filenames, val_label_filenames, BATCH_SIZE)
    # Build a Graph that computes the logits predictions from the
    # inference model.

    #loss, eval_prediction = model.inference(train_data_node, train_labels_node, phase_train)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #train_op = train(loss, global_step)

    #createLogs()


def get_filename_list(path):
  #fd = open(path)

  fileNames = os.listdir(path)
  sorted_filenames = sorted(fileNames) #sort by names to get img and label after each other

  image_filenames = []
  label_filenames = []
  filenames = []
  for i in range (0, len(sorted_filenames), 2):
    image_filenames.append(sorted_filenames[i])
    label_filenames.append(sorted_filenames[i+1])
  return image_filenames, label_filenames

def datasetInputs(image_filenames, label_filenames, batch_size): #prev name: camVidInputs

  images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
  labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

  filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

  image, label = dataset_reader(filename_queue)
  reshaped_image = tf.cast(image, tf.float32)

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d input images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def dataset_reader(filename_queue): #prev name: CamVid_reader

 image_filename = filename_queue[0] #tensor of type string
 label_filename = filename_queue[1] #tensor of type string

 #get png encoded image
 imageValue = tf.read_file(image_filename)
 labelValue = tf.read_file(label_filename)

 #decodes a png image into a uint8 or uint16 tensor
 #returns a tensor of type dtype with shape [height, width, depth]
 image_bytes = tf.image.decode_png(imageValue)
 label_bytes = tf.image.decode_png(labelValue)

 image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
 label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

 return image, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 3-D Tensor of [height, width, 1] type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, label_batch


if __name__ == "__main__":
  print('in main')
  run()
