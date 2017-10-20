from .inference import inference_basic, inference_basic_dropout, inference_extended_dropout, inference_extended
from .training import training
from .test import test
from .inputs import placeholder_inputs, dataset_inputs, get_filename_list, get_all_test_data
from .evaluation import evaluation, loss_calc, per_class_acc, get_hist, print_hist_summery

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

""" AFFECTS HOW CODE RUNS"""

tf.app.flags.DEFINE_string('model', 'basic_dropout',
                            """ Defining what version of the model to run """)

#Training
tf.app.flags.DEFINE_string('log_dir',"./tmp/basic_dropout", #Training is default on, unless testing or finetuning is set to "True"
                           """ dir to store training ckpt """)
tf.app.flags.DEFINE_integer('max_steps', "60000",
                            """ max_steps for training """)

#Testing
tf.app.flags.DEFINE_boolean('testing', False, #True or False
                            """ Whether to run test or not """)
tf.app.flags.DEFINE_string('model_ckpt_dir', "./tmp/basic_dropout/model.ckpt-22500",
                           """ checkpoint file for model to use for testing """)
tf.app.flags.DEFINE_boolean('save_image', True,
                            """ Whether to save predicted image """)
tf.app.flags.DEFINE_string('res_output_dir', "/home/mators/autoKart/result_imgs",
                            """ Directory to save result images when running test """)
#Finetuning
tf.app.flags.DEFINE_boolean('finetune', True, #True or False
                           """ Whether to finetune or not """)
tf.app.flags.DEFINE_string('finetune_dir', 'tmp/basic_dropout/model.ckpt-22500',
                           """ Path to the checkpoint file to finetune from """)


""" TRAINING PARAMETERS"""
tf.app.flags.DEFINE_integer('batch_size', "6",
                            """ train batch_size """)
tf.app.flags.DEFINE_integer('test_batch_size', "1",
                            """ batch_size for training """)
tf.app.flags.DEFINE_integer('eval_batch_size', "6",
                            """ Eval batch_size """)

tf.app.flags.DEFINE_float('balance_weight_0', 0.8,
                            """ Define the dataset balance weight for class 0 - Not building """)
tf.app.flags.DEFINE_float('balance_weight_1', 1.1,
                            """ Define the dataset balance weight for class 1 - Building """)


""" DATASET SPECIFIC PARAMETERS """
#Directories
tf.app.flags.DEFINE_string('train_dir', "/home/mators/aerial_datasets/RGB_Trondheim_full/RGB_images/combined_dataset_v2/train_images",
                           """ path to training images """)
tf.app.flags.DEFINE_string('test_dir', "/home/mators/aerial_datasets/RGB_Trondheim_full/RGB_images/combined_dataset_v2/test_images",
                           """ path to test image """)
tf.app.flags.DEFINE_string('val_dir', "/home/mators/aerial_datasets/RGB_Trondheim_full/RGB_images/combined_dataset_v2/val_images",
                           """ path to val image """)

#Dataset size. #Epoch = one pass of the whole dataset.
tf.app.flags.DEFINE_integer('num_examples_epoch_train', "7121",
                           """ num examples per epoch for train """)
tf.app.flags.DEFINE_integer('num_examples_epoch_test', "889",
                           """ num examples per epoch for test """)
tf.app.flags.DEFINE_integer('num_examples_epoch_val', "50",
                           """ num examples per epoch for test """)
tf.app.flags.DEFINE_float('fraction_of_examples_in_queue', "0.1",
                           """ Fraction of examples from datasat to put in queue. Large datasets need smaller value, otherwise memory gets full. """)

#Image size and classes
tf.app.flags.DEFINE_integer('image_h', "512",
                            """ image height """)
tf.app.flags.DEFINE_integer('image_w', "512",
                            """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3",
                            """ number of image channels (RGB) (the depth) """)
tf.app.flags.DEFINE_integer('num_class', "2", #classes are "Building" and "Not building"
                            """ total class number """)


#FOR TESTING:
TEST_ITER = FLAGS.num_examples_epoch_test // FLAGS.batch_size


tf.app.flags.DEFINE_float('moving_average_decay', "0.99",#"0.9999", #https://www.tensorflow.org/versions/r0.12/api_docs/python/train/moving_averages
                           """ The decay to use for the moving average""")


if(FLAGS.model == "basic" or FLAGS.model == "basic_dropout"):
    tf.app.flags.DEFINE_string('conv_init', 'xavier', # xavier / var_scale
                            """ Initializer for the convolutional layers. One of: "xavier", "var_scale".  """)
    tf.app.flags.DEFINE_string('optimizer', "SGD",
                            """ Optimizer for training. One of: "adam", "SGD", "momentum", "adagrad". """)

elif(FLAGS.model == "extended" or FLAGS.model == "extended_dropout"):
    tf.app.flags.DEFINE_string('conv_init', 'var_scale', # xavier / var_scale
                            """ Initializer for the convolutional layers. One of "msra", "xavier", "var_scale".  """)
    tf.app.flags.DEFINE_string('optimizer', "adagrad",
                            """ Optimizer for training. One of: "adam", "SGD", "momentum", "adagrad". """)
else:
    raise ValueError("Determine which initalizer you want to use. Non exist for model ", FLAGS.model)