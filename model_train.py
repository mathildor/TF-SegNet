
import tensorflow as tf

#rest of the code
import model
import Utils
import Inputs


BATCH_SIZE = 1
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.


def train():
  """ Train model a number of steps """


  batch_size = BATCH_SIZE
  train_dir = "./tmp/logs"
  #image_filenames, label_filenames = get_filename_list("tmp3/first350/SegNet-Tutorial/CamVid/train.txt")
  image_filenames, label_filenames = Inputs.get_filename_list("../dataset/CamVid/train.txt")
  #image_filenames, label_filenames = Inputs.get_filename_list("./dataset/dummy_set/train/images")
  #val_image_filenames, val_label_filenames = get_filename_list("tmp3/first350/SegNet-Tutorial/CamVid/val.txt")
  val_image_filenames, val_label_filenames = Inputs.get_filename_list("../dataset/CamVid/val.txt")
  #val_image_filenames, val_label_filenames = Inputs.get_filename_list("./dataset/dummy_set/val/images")

  with tf.Graph().as_default():

    train_data_node = tf.placeholder(
          tf.float32,
          shape=[batch_size, 360, 480, 3])

    train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, 360, 480, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    global_step = tf.Variable(0, trainable=False)

    # For CamVid
      #Make images into correct type(float32/float16 el.), create shuffeled batches ++
    images, labels = Inputs.datasetInputs(image_filenames, label_filenames, BATCH_SIZE) #prev name for datasetInputs = CamVidInputs

    val_images, val_labels = Inputs.datasetInputs(val_image_filenames, val_label_filenames, BATCH_SIZE)
    # Build a Graph that computes the logits predictions from the
    # inference model.

    loss, eval_prediction = model.inference(train_data_node, train_labels_node, phase_train)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # The op for initializing the variables.
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    createLogs(loss, eval_prediction, val_images, val_labels, train_dir, images, labels)



def createLogs(loss, eval_prediction, val_images, val_labels, train_dir, images, labels): #lagt til funksjonen selv, usikker på navnet
    max_steps = 20000

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()

      # Start running operations on the Graph.
      sess.run(init)

      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
      average_pl = tf.placeholder(tf.float32)
      acc_pl = tf.placeholder(tf.float32)
      iu_pl = tf.placeholder(tf.float32)
      average_summary = tf.summary.scalar("test_average_loss", average_pl)
      acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
      iu_summary = tf.summary.scalar("Mean_IU", iu_pl)
      for step in range(max_steps):
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
          2xamples_per_step = batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch pre-class accuracy
          pred = sess.run(eval_prediction, feed_dict=feed_dict)
          per_class_acc(eval_batches(image_batch, sess, eval_prediction=pred), label_batch)

        if step % 100 == 0:
          print("start testing.....")
          total_val_loss = 0.0
          hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
          for test_step in range(TEST_ITER):
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

            _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
              train_data_node: val_images_batch,
              train_labels_node: val_labels_batch,
              phase_train: True
            })
            total_val_loss += _val_loss
            hist += get_hist(_val_pred, val_labels_batch)
          print("val loss: ", total_val_loss / TEST_ITER)
          acc_total = np.diag(hist).sum() / hist.sum()
          iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
          test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
          acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
          iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
          print_hist_summery(hist)
          # per_class_acc(eval_batches(val_images_batch, sess, eval_prediction=_val_pred), val_labels_batch)

          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.add_summary(test_summary_str, step)
          summary_writer.add_summary(acc_summary_str, step)
          summary_writer.add_summary(iu_summary_str, step)
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
          checkpoint_path = os.path.join(train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

      coord.request_stop()
      coord.join(threads)


def test():
  #checkpoint_dir = "tmp4/first350/TensorFlow/Logs"
  # testing should set BATCH_SIZE = 1
  batch_size = 1

  # image_filenames, label_filenames = Inputs.get_filename_list("./dataset/dummy_set/val/images") #just for testing, should use diff images
  image_filenames, label_filenames = Inputs.get_filename_list("../dataset/CamVid/test.txt") #just for testing, should use diff images

  test_data_node = tf.placeholder(
        tf.float32,
        shape=[batch_size, 360, 480, 3])

  test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, 360, 480, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  loss, logits = model.inference(test_data_node, test_labels_node, phase_train)

  # pred = tf.argmax(logits, dimension=3)

  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
                      MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, "tmp4/first350/TensorFlow/Logs/model.ckpt-" )
    images, labels = get_all_test_data(image_filenames, label_filenames)
    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for image_batch, label_batch  in zip(images, labels):
      print(image_batch.shape, label_batch.shape)
      feed_dict = {
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False
      }
      dense_prediction = sess.run(logits, feed_dict=feed_dict)
      print(dense_prediction.shape)
      hist += get_hist(dense_prediction, label_batch)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))


if __name__ == "__main__":

  #tf.app.run() is called in cifar code - wrapper that handles flags parsing. Should it be added?
  #test() what dose this do? looks like it loads prev checkpoints
  #exit()

  with tf.device('/gpu:3'):
    train()
