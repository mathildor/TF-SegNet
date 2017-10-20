import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

def training(loss):

    global_step = tf.Variable(0, name='global_step', trainable=False)

    #This motif is needed to hook up the batch_norm updates to the training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if(FLAGS.optimizer == "SGD"):
            print("Running with SGD optimizer")
            optimizer = tf.train.GradientDescentOptimizer(0.1)
        elif(FLAGS.optimizer == "adam"):
            print("Running with adam optimizer")
            optimizer = tf.train.AdamOptimizer(0.001)
        elif(FLAGS.optimizer == "adagrad"):
            print("Running with adagrad optimizer")
            optimizer = tf.train.AdagradOptimizer(0.01)
        else:
            raise ValueError("optimizer was not recognized.")

        train_op = optimizer.minimize(loss=loss, global_step=global_step)
        #optimizer, like 'SGD', 'Adam', 'Adagrad'
        #train_op = tf.contrib.layers.optimize_loss(loss, optimizer="SGD", global_step=global_step, learning_rate = 0.1)
    return train_op, global_step