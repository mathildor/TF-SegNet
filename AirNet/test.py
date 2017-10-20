import tensorflow as tf
import AirNet
import os
import numpy as np
from PIL import Image
FLAGS = tf.app.flags.FLAGS


def test():
    print("----------- In test method ----------")
    image_filenames, label_filenames = AirNet.get_filename_list(FLAGS.test_dir)

    test_data_node, test_labels_node, is_training, keep_prob = AirNet.placeholder_inputs(batch_size=1)  
    

    if FLAGS.model == "basic":
        logits = AirNet.inference_basic(test_data_node, is_training)
    elif FLAGS.model == "extended":
        logits = AirNet.inference_extended(test_data_node, is_training)
    elif FLAGS.model == "basic_dropout":
        logits = AirNet.inference_basic_dropout(test_data_node, is_training, keep_prob)
    elif FLAGS.model == "extended_dropout":
        logits = AirNet.inference_extended_dropout(test_data_node, is_training, keep_prob)
    else:
        raise ValueError("The selected model does not exist")

    pred = tf.argmax(logits, axis=3)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.model_ckpt_dir)
        images, labels = AirNet.get_all_test_data(image_filenames, label_filenames)
        
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
        hist = np.zeros((FLAGS.num_class, FLAGS.num_class))
        
        step=0
        for image_batch, label_batch  in zip(images, labels):
            feed_dict = {
                test_data_node: image_batch,
                test_labels_node: label_batch,
                is_training: False,
                keep_prob: 1.0 #During testing droput should be turned off -> 100% chance of keeping variable
            }

            dense_prediction, im = sess.run(fetches=[logits, pred], feed_dict=feed_dict)
            AirNet.per_class_acc(dense_prediction, label_batch)
            # output_image to verify
            if (FLAGS.save_image):
                if(step < 10):
                    numb_img = "000"+str(step)
                elif(step < 100):
                    numb_img = "00"+str(step)
                elif(step < 1000):
                    numb_img = "0"+str(step)
                write_image(im[0], os.path.join(FLAGS.res_output_dir +'/testing_image'+numb_img+'.png')) #Printing all test images
            step=step+1
            hist += AirNet.get_hist(dense_prediction, label_batch)
        acc_total = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("acc: ", acc_total)
        print("IU: ", iu)
        print("mean IU: ", np.nanmean(iu))
        
        coord.request_stop()
        coord.join(threads)


def write_image(image, filename):
    """ store label data to colored image """
    Sky = [0,0,0] #
    Building = [128,128,0] #green-ish

    r = image.copy()
    g = image.copy()
    b = image.copy()

    label_colours = np.array([Sky, Building])
    for label in range(0,FLAGS.num_class): #for all labels - shouldn't this be set according to num_class?
        #Replacing all instances in matrix with label value with the label colour
        r[image==label] = label_colours[label,0] #red is channel/debth 0
        g[image==label] = label_colours[label,1] #green is channel/debth 1
        b[image==label] = label_colours[label,2] #blue is channel/debth 2
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)