
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import os

#other modules
import model
import Utils
import Inputs


FLAGS = tf.app.flags.FLAGS

""" AFFECTS HOW CODE RUNS"""

#Training
tf.app.flags.DEFINE_string('log_dir', "tmp/IR_RGB_0.1res_IR_set_v2/test/logs",
                           """ dir to store training ckpt """)
tf.app.flags.DEFINE_integer('max_steps', "20000",
                            """ max_steps for training """)

#Testing
tf.app.flags.DEFINE_boolean('testing', False, #insert path to log file: tmp/logs/model.ckpt-19999.
                            """ Whether to run test or not """)
tf.app.flags.DEFINE_string('model_ckpt_dir', 'tmp/IR_RGB_0.1res_IR_set_v2/dropout/logs/model.ckpt-22500', #insert path to log file: tmp/logs/model.ckpt-19999.
                           """ checkpoint file for model to use for testing """)
tf.app.flags.DEFINE_boolean('save_image', True,
                            """ Whether to save predicted image """)

#Finetunings
tf.app.flags.DEFINE_boolean('finetune', True,
                           """ Whether to finetune or not """)
tf.app.flags.DEFINE_string('finetune_dir', 'tmp/IR_RGB_0.1res_IR_set_v2/dropout/logs/model.ckpt-4999',
                           """ Path to the checkpoint file to finetune from """)


"""DATASET SPECIFIC PARAMETERS"""
#Image size
tf.app.flags.DEFINE_integer('image_h', "512",
                            """ image height """)
tf.app.flags.DEFINE_integer('image_w', "512",
                            """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3",
                            """ number image channels (RGB) (the depth) """)


#Directories
# tf.app.flags.DEFINE_string('image_dir', "../aerial_img_4600/train_images/png",
#                            """ path to training images """)
# tf.app.flags.DEFINE_string('test_dir', "../aerial_img_4600/test_images/png",
#                            """ path to test image """)
# tf.app.flags.DEFINE_string('val_dir', "../aerial_img_4600/val_images/png",
#                            """ path to val image """)
tf.app.flags.DEFINE_string('image_dir', "../aerial_datasets/IR_RGB_0.1res/IR_images/combined_dataset_v2/train_images",
                           """ path to training images """)
tf.app.flags.DEFINE_string('test_dir', "../aerial_datasets/IR_RGB_0.1res/IR_images/combined_dataset_v2/test_images",
                           """ path to test image """)
tf.app.flags.DEFINE_string('val_dir', "../aerial_datasets/IR_RGB_0.1res/IR_images/combined_dataset_v2/val_images",
                           """ path to val image """)

#Dataset size. #Epoch = one pass of the whole dataset.
tf.app.flags.DEFINE_integer('num_examples_epoch_train', "3720",
                           """ num examples per epoch for train """)
tf.app.flags.DEFINE_integer('num_examples_epoch_test', "460",
                           """ num examples per epoch for test """)
tf.app.flags.DEFINE_float('fraction_of_examples_in_queue', "0.4",
                           """ Fraction of examples from datasat to put in queue. Large datasets need smaller value, otherwise memory gets full. """)

tf.app.flags.DEFINE_integer('num_class', "2", #classes are "Building" and "Not building"
                            """ total class number """)

""" TRAINING PARAMETERS"""
tf.app.flags.DEFINE_integer('batch_size', "5",
                            """ batch_size """)
tf.app.flags.DEFINE_integer('test_batch_size', "1",
                            """ batch_size for training """)
tf.app.flags.DEFINE_integer('eval_batch_size', "6",
                            """ Eval batch_size """)


tf.app.flags.DEFINE_float('learning_rate', "1e-3", #Figure out what is best for AdamOptimizer!
                           """ initial lr """)

tf.app.flags.DEFINE_float('moving_average_decay', "0.9999", #https://www.tensorflow.org/versions/r0.12/api_docs/python/train/moving_averages
                           """ The decay to use for the moving average""")




#FOR TESTING:
TEST_ITER = FLAGS.num_examples_epoch_test // FLAGS.batch_size
PREV_VAL_LOSS = 0 #Used to print loss for previous validation batch to look for overfitting

def train(is_finetune=False):
  """ Train model a number of steps """


  # should be changed if model is stored by different convention
  startstep = 0 if not is_finetune else int(FLAGS.finetune_dir.split('-')[-1])
  image_filenames, label_filenames = Inputs.get_filename_list(FLAGS.image_dir)
  val_image_filenames, val_label_filenames = Inputs.get_filename_list(FLAGS.val_dir)

  with tf.Graph().as_default():
    #Probablitity that the neuron's output will be kept during dropout
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    global_step = tf.Variable(0, trainable=False)

    #Make images into correct type(float32/float16 el.), create shuffeled batches ++
    images, labels = Inputs.datasetInputs(image_filenames, label_filenames, FLAGS.batch_size)
    val_images, val_labels = Inputs.datasetInputs(val_image_filenames, val_label_filenames, FLAGS.batch_size)

    train_data_node = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_h, FLAGS.image_w, 3])
    train_labels_node = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, FLAGS.image_h, FLAGS.image_w, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # Build a Graph that computes the logits predictions from the inference model.
    logits = model.inference(train_data_node, phase_train, FLAGS.batch_size, keep_probability) #tensor, nothing calculated yet
    print("\n -- After logts init!")

    #Calculate loss:
    loss = model.cal_loss(logits, train_labels_node)
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = model.train(loss, global_step)
    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    #Using a context manager - it will release resources for session when no longer required.
    #Defining session like this means you do not have to explicitly close the session.
    with tf.Session() as sess:
      if (is_finetune == True):
          saver.restore(sess, FLAGS.finetune_dir)
      else:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())

      print("Started session run")
      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
      average_pl = tf.placeholder(tf.float32)
      acc_pl = tf.placeholder(tf.float32)
      iu_pl = tf.placeholder(tf.float32)
      average_summary = tf.summary.scalar("test_average_loss", average_pl)
      acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
      iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

      """ Starting iterations to train the network """
      for step in range(startstep, startstep + FLAGS.max_steps):
        image_batch ,label_batch = sess.run(fetches=[images, labels])
        # since we still use mini-batches in eval, still set bn-layer phase_train = True
        feed_dict = {
          train_data_node: image_batch,
          train_labels_node: label_batch,
          phase_train: True,
          keep_probability: 0.5
        }
        # storeImageQueue(image_batch, label_batch, step)
        start_time = time.time()

        _, loss_value = sess.run(fetches=[train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN - weights have "exploded"'

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
          print('\n--- Normal training ---')
          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch pre-class accuracy
          pred = sess.run(fetches=logits, feed_dict=feed_dict)
          print("\n -- conv in classifier!")

          Utils.per_class_acc(pred, label_batch)

        if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
          """ Validate training by running validation dataset """
          print(" \n --- Starting testing by running validation dataset ---")
          total_val_loss = 0.0
          hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
          for test_step in range(TEST_ITER):
            val_images_batch, val_labels_batch = sess.run(fetches=[val_images, val_labels])
            feed_dict = {
              train_data_node: val_images_batch,
              train_labels_node: val_labels_batch,
              phase_train: True,
              keep_probability: 1.0 #During validation training no dropout layers
            }
            _val_loss, _val_pred = sess.run(fetches=[loss, logits], feed_dict=feed_dict)
            total_val_loss += _val_loss
            hist += Utils.get_hist(_val_pred, val_labels_batch)
          PREV_VAL_LOSS = total_val_loss/TEST_ITER
          print("loss for validation dataset: ", total_val_loss / TEST_ITER, ". If this value increases the model is likely overfitting.") #Prev value was: ", PREV_VAL_LOSS)
          writeSummaries(sess, hist, average_summary, average_pl, total_val_loss, acc_summary, iu_summary, acc_pl, iu_pl, summary_op, step, feed_dict, summary_writer)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or step % 500 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

      coord.request_stop()
      coord.join(threads)


def writeSummaries(sess, hist,average_summary, average_pl, total_val_loss, acc_summary, iu_summary, acc_pl, iu_pl, summary_op, step, feed_dict, summary_writer):
  acc_total = np.diag(hist).sum() / hist.sum()
  iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
  test_summary_str = sess.run(fetches=average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
  acc_summary_str = sess.run(fetches=acc_summary, feed_dict={acc_pl: acc_total})
  iu_summary_str = sess.run(fetches=iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
  Utils.print_hist_summery(hist)
  # per_class_acc(model.eval_batches(val_images_batch, sess, eval_prediction=_val_pred), val_labels_batch)

  summary_str = sess.run(fetches=summary_op, feed_dict=feed_dict)
  summary_writer.add_summary(summary_str, step)
  summary_writer.add_summary(test_summary_str, step)
  summary_writer.add_summary(acc_summary_str, step)
  summary_writer.add_summary(iu_summary_str, step)

def test():
  print("----------- In test method ----------")

  testing_batch_size = 1

  keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

  image_filenames, label_filenames = Inputs.get_filename_list(FLAGS.test_dir)
  test_data_node = tf.placeholder(tf.float32, shape=[testing_batch_size, FLAGS.image_h, FLAGS.image_w, FLAGS.image_c])  #360, 480, 3
  test_labels_node = tf.placeholder(tf.int64, shape=[FLAGS.test_batch_size, FLAGS.image_h, FLAGS.image_w, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  logits = model.inference(test_data_node, phase_train, testing_batch_size, keep_probability)
  loss = model.cal_loss(logits, test_labels_node)
  pred = tf.argmax(logits, dimension=3)

  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, FLAGS.model_ckpt_dir)

    images, labels = Inputs.get_all_test_data(image_filenames, label_filenames)
    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
    step=0
    for image_batch, label_batch  in zip(images, labels):
      feed_dict = { #maps graph elements to values
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False,
        keep_probability: 1.0 #During testing droput should be turned off -> 100% chance for keeping variable
      }

      dense_prediction, im = sess.run(fetches=[logits, pred], feed_dict=feed_dict)
      #Testing if this can be added
      Utils.per_class_acc(dense_prediction, label_batch)
      # output_image to verify
      if (FLAGS.save_image):
          Utils.writeImage(im[0], os.path.join('result_imgs','testing_image'+str(step)+'.png')) #Printing all test images
      step=step+1
      hist += Utils.get_hist(dense_prediction, label_batch)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))


def checkArgs():
    if FLAGS.testing == True:
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.model_ckpt_dir)
        print("Dataset testing dir: %s"%FLAGS.test_dir)
    elif FLAGS.finetune == True:
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("Dataset Image dir: %s"%FLAGS.image_dir)
        print("Dataset Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("Dataset Image dir: %s"%FLAGS.image_dir)
        print("Dataset Val dir: %s"%FLAGS.val_dir)

    print("Batch Size: %d"%FLAGS.batch_size)
    print("Log dir: %s"%FLAGS.log_dir)

def main(args):
    checkArgs()
    if FLAGS.testing:
        print("Testing the model!")
        test()
    elif FLAGS.finetune:
        print("Finetuning the model!")
        train(is_finetune=True)
    else:
        print("Training from scratch!")
        train(is_finetune=False)

if __name__ == "__main__":
    tf.app.run() # wrapper that handles flags parsing.
