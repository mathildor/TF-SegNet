import os
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import AirNet 

FLAGS = tf.app.flags.FLAGS

def train(is_finetune = False):
    
    startstep = 0 if not is_finetune else int(FLAGS.finetune_dir.split('-')[-1])
    image_filenames, label_filenames = AirNet.get_filename_list(FLAGS.train_dir)
    val_image_filenames, val_label_filenames = AirNet.get_filename_list(FLAGS.val_dir)

    with tf.Graph().as_default():

        images, labels, is_training, keep_prob = AirNet.placeholder_inputs(batch_size=FLAGS.batch_size)

        images, labels = AirNet.dataset_inputs(image_filenames, label_filenames, FLAGS.batch_size)
        val_images, val_labels = AirNet.dataset_inputs(val_image_filenames, val_label_filenames, FLAGS.eval_batch_size, False)

        if FLAGS.model == "basic":
            logits = AirNet.inference_basic(images, is_training)
        elif FLAGS.model == "extended":
             logits = AirNet.inference_extended(images, is_training)
        elif FLAGS.model == "basic_dropout":
            logits = AirNet.inference_basic_dropout(images, is_training, keep_prob)
        elif FLAGS.model == "extended_dropout":
            logits = AirNet.inference_extended_dropout(images, is_training, keep_prob)
        else:
            raise ValueError("The selected model does not exist")

        loss = AirNet.loss_calc(logits=logits, labels=labels)
        train_op, global_step = AirNet.training(loss=loss)
        accuracy = tf.argmax(logits, axis=3)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=100000)
        
        with tf.Session() as sess:

            if(is_finetune):
                print("\n =====================================================")
                print("  Finetuning with model: ", FLAGS.model)
                print("\n    Batch size is: ", FLAGS.batch_size)
                print("    ckpt files are saved to: ", FLAGS.log_dir)
                print("    Max iterations to train is: ", FLAGS.max_steps)
                print(" =====================================================")
                saver.restore(sess, FLAGS.finetune_dir)
            else:
                print("\n =====================================================")
                print("  Training from scratch with model: ", FLAGS.model)
                print("\n    Batch size is: ", FLAGS.batch_size)
                print("    ckpt files are saved to: ", FLAGS.log_dir)
                print("    Max iterations to train is: ", FLAGS.max_steps)
                print(" =====================================================")
                sess.run(tf.variables_initializer(tf.global_variables()))
                sess.run(tf.local_variables_initializer())

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            #val_writer = tf.summary.FileWriter(#TEST_WRITER_DIR)

            """ Starting iterations to train the network """
            for step in range(startstep+1, startstep + FLAGS.max_steps+1):
                images_batch, labels_batch = sess.run(fetches=[images, labels])

                train_feed_dict = {images: images_batch,
                                   labels: labels_batch,
                                   is_training: True, 
                                   keep_prob: 0.5}

                start_time = time.time()
                    
                _, train_loss_value, train_accuracy_value, train_summary_str = sess.run([train_op, loss, accuracy, summary], feed_dict=train_feed_dict)
                
                #Finding duration for training batch
                duration = time.time() - start_time

                if step % 10 == 0: #Print info about training                   
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = float(duration)

                    print('\n--- Normal training ---')
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, train_loss_value,
                                         examples_per_sec, sec_per_batch))

                    # eval current training batch pre-class accuracy
                    pred = sess.run(logits, feed_dict=train_feed_dict)
                    AirNet.per_class_acc(pred, labels_batch) #printing class accuracy

                    train_writer.add_summary(train_summary_str, step)
                    train_writer.flush()
                
                if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                    test_iter = FLAGS.num_examples_epoch_test // FLAGS.batch_size
                    """ Validate training by running validation dataset """ 
                    print("\n===========================================================")
                    print("--- Running test on VALIDATION dataset ---")
                    total_val_loss=0.0
                    hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
                    for val_step in range (test_iter):
                        val_images_batch, val_labels_batch = sess.run(fetches=[val_images, val_labels])

                        val_feed_dict = { images: val_images_batch,
                                          labels: val_labels_batch,
                                          is_training: True,
                                          keep_prob: 1.0}

                        _val_loss, _val_pred = sess.run(fetches=[loss, logits], feed_dict=val_feed_dict)
                        total_val_loss += _val_loss
                        hist += AirNet.get_hist(_val_pred, val_labels_batch)
                    print("Validation Loss: ", total_val_loss / test_iter, ". If this value increases the model is likely overfitting.")
                    AirNet.print_hist_summery(hist)
                    print("===========================================================")

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or step % 500 == 0 or (step + 1) == FLAGS.max_steps:
                    print("\n--- SAVING SESSION ---")
                    checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print("=========================")
            
            coord.request_stop()
            coord.join(threads)

def main(args):
    if FLAGS.testing:
        print("Testing the model!")
        AirNet.test()
    elif FLAGS.finetune:
        train(is_finetune=True)
    else:
        train(is_finetune=False)

if __name__ == "__main__":
    tf.app.run() # wrapper that handles flags parsing.