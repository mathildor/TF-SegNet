import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS

def loss_calc(logits, labels):
    """
        logits: tensor, float - [batch_size, width, height, num_classes].
        labels: tensor, int32 - [batch_size, width, height, num_classes].
    """
    # construct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    labels = tf.reshape(tf.one_hot(label_flat, depth=FLAGS.num_class), (-1, FLAGS.num_class))

    #This motif is needed to hook up the batch_norm updates to the training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar('loss', cross_entropy)
    return cross_entropy

def weighted_loss_calc(logits, labels):
    class_weights = np.array([
        FLAGS.balance_weight_0, #"Not building"
        FLAGS.balance_weight_1 #"Building"
    ])
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=labels, pos_weight=class_weights)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
    return loss

def evaluation(logits, labels):
    labels = tf.to_int64(labels)
    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    num_class = FLAGS.num_class
    size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
      hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f'%np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f'%np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0
        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f "%(ii,acc))

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_hist(predictions, labels):
    num_class = predictions.shape[3] #becomes 2 for aerial - correct
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist

def print_hist_summery(hist):
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f'%np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f'%np.nanmean(iu))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f "%(ii, acc))