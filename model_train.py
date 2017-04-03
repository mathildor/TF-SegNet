
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import os

#rest of the code
import model
import Utils
import Inputs


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('testing', 'tmp/logs/model.ckpt-19999', #insert path to log file: tmp/logs/model.ckpt-19999. Running automatic if not empty string
                           """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '',
                           """ finetune checkpoint file """)
tf.app.flags.DEFINE_boolean('save_image', True,
                            """ whether to save predicted image """)


tf.app.flags.DEFINE_integer('batch_size', "5",
                            """ batch_size """)
tf.app.flags.DEFINE_integer('test_batch_size', "1",
                            """ batch_size for training """)
tf.app.flags.DEFINE_integer('eval_batch_size', "5",
                            """ Eval batch_size """)


tf.app.flags.DEFINE_float('learning_rate', "1e-3", #Figure out what is best for AdamOptimizer!
                           """ initial lr """)

tf.app.flags.DEFINE_float('moving_average_decay', "0.9999",
                           """ The decay to use for the moving average""")

tf.app.flags.DEFINE_integer('max_steps', "20000",
                            """ max_steps """)
tf.app.flags.DEFINE_integer('num_class', "11",
                            """ total class number """)

""" Image size """
tf.app.flags.DEFINE_integer('image_h', "360",
                            """ image height """)
tf.app.flags.DEFINE_integer('image_w', "480",
                            """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3",
                            """ number image channels (RGB) (the depth) """)

""" Directories  """
tf.app.flags.DEFINE_string('log_dir', "tmp1/logs",
                           """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "../SegNet/CamVid/train.txt",
                           """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "../SegNet/CamVid/test.txt",
                           """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "../SegNet/CamVid/val.txt",
                           """ path to CamVid val image """)




#FOR TESTING:
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST // FLAGS.batch_size



def train(is_finetune=False): #Now set to
  """ Train model a number of steps """

  # should be changed if your model stored by different conventio
  startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

  image_filenames, label_filenames = Inputs.get_filename_list(FLAGS.image_dir)
  val_image_filenames, val_label_filenames = Inputs.get_filename_list(FLAGS.val_dir)

  with tf.Graph().as_default():

    train_data_node = tf.placeholder(
          tf.float32,
          shape=[FLAGS.batch_size, 360, 480, 3])

    train_labels_node = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, 360, 480, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    global_step = tf.Variable(0, trainable=False)

    # For CamVid
      #Make images into correct type(float32/float16 el.), create shuffeled batches ++
    images, labels = Inputs.datasetInputs(image_filenames, label_filenames, FLAGS.batch_size) #prev name for datasetInputs = CamVidInputs

    val_images, val_labels = Inputs.datasetInputs(val_image_filenames, val_label_filenames, FLAGS.batch_size)


    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(train_data_node, phase_train, FLAGS.batch_size)

    #Calculate loss:
    loss = model.cal_loss(logits, train_labels_node)


    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer() )


    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
      # Build an initialization operation to run below.
      if (is_finetune == True):
          saver.restore(sess, FLAGS.testing ) #FLAGS.testing is ckpt
      else:
          init = tf.global_variables_initializer()
          sess.run(init)

      # Start running operations on the Graph.
      sess.run(init)

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
        image_batch ,label_batch = sess.run([images, labels])
        # since we still use mini-batches in eval, still set bn-layer phase_train = True
        feed_dict = {
          train_data_node: image_batch,
          train_labels_node: label_batch,
          phase_train: True
        }
        # storeImageQueue(image_batch, label_batch, step)
        start_time = time.time()

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch pre-class accuracy
          pred = sess.run(logits, feed_dict=feed_dict)
          model.per_class_acc(model.eval_batches(image_batch, sess, eval_prediction=pred), label_batch)

        if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
          print("start testing.....")
          total_val_loss = 0.0
          hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
          for test_step in range(TEST_ITER): #TEST_ITER is a number
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

            _val_loss, _val_pred = sess.run([loss, logits], feed_dict={
              train_data_node: val_images_batch,
              train_labels_node: val_labels_batch,
              phase_train: True
            })
            total_val_loss += _val_loss
            hist += model.get_hist(_val_pred, val_labels_batch, FLAGS.batch_size)
          print("val loss: ", total_val_loss / TEST_ITER)
          acc_total = np.diag(hist).sum() / hist.sum()
          iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
          test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
          acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
          iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
          model.print_hist_summery(hist)
          # per_class_acc(model.eval_batches(val_images_batch, sess, eval_prediction=_val_pred), val_labels_batch)

          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.add_summary(test_summary_str, step)
          summary_writer.add_summary(acc_summary_str, step)
          summary_writer.add_summary(iu_summary_str, step)
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

      coord.request_stop()
      coord.join(threads)


def test():
  print("----------- !!!!!!!!!!in test method!!!!!!!!!!!!!! ----------")

  testing_batch_size = 1

  image_filenames, label_filenames = Inputs.get_filename_list(FLAGS.test_dir)
  test_data_node = tf.placeholder(
        tf.float32,
        shape=[testing_batch_size, FLAGS.image_h, FLAGS.image_w, FLAGS.image_c])  #360, 480, 3
  test_labels_node = tf.placeholder(tf.int64, shape=[FLAGS.test_batch_size, FLAGS.image_h, FLAGS.image_w, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  logits = model.inference(test_data_node, phase_train, testing_batch_size)

  #Calculate loss:
  loss = model.cal_loss(logits, test_labels_node) #HER KRÆSJER!

  pred = tf.argmax(logits, dimension=3)

  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
                      FLAGS.moving_average_decay)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, FLAGS.testing) #originally said: "tmp4/first350/TensorFlow/Logs/model.ckpt-"

    images, labels = Inputs.get_all_test_data(image_filenames, label_filenames)

    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
    for image_batch, label_batch  in zip(images, labels):
      feed_dict = {
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False
      }

      dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
      #print(dense_prediction.shape)

      # output_image to verify
      if (FLAGS.save_image):
          Utils.writeImage(im[0], 'testing_image.png')

      hist += model.get_hist(dense_prediction, label_batch, testing_batch_size)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))


def checkArgs():
    if FLAGS.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.testing)
        print("CamVid testing dir: %s"%FLAGS.test_dir)
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)

    print("Batch Size: %d"%FLAGS.batch_size)
    print("Log dir: %s"%FLAGS.log_dir)

def main(args):
    checkArgs()
    if FLAGS.testing:
        test()
    elif FLAGS.finetune:
        print("Finetuning the model!")
        train(is_finetune=True)
    else:
        print("Training from scratch!")
        train(is_finetune=False)

if __name__ == "__main__":
    tf.app.run() # wrapper that handles flags parsing.
