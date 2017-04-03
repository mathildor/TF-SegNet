import tensorflow as tf
from PIL import Image
import numpy as np

FLAGS = tf.app.flags.FLAGS


def writeImage(image, filename):
    """ store label data to colored image """
    Sky = [128,128,128] #sky
    Building = [128,0,0] #dark red
    Pole = [192,192,128] #light yellow
    Road_marking = [255,69,0] #bright orange
    Road = [128,64,128] #purple
    Pavement = [60,40,222]#bright blue
    r = image.copy()
    g = image.copy()
    b = image.copy()
    print (r[0])
    testImg=[[1, 1, 1], [2, 2, 2]]
    label_colours = np.array([Sky, Building, Pole, Road, Road_marking, Pavement])
    for label in range(0,6): #for all labels - shouldn't this be set according to num_class
        #Replacing all instances in matrix with label value with the label colour
        r[image==label] = label_colours[label,0] #red is channel/debth 0
        print('AFTER------r')
        print (r[0])
        #testImg[image==label] = label_colours[label,0] #red is channel/debth 0
        print(testImg)
        sys.out
        g[image==label] = label_colours[label,1] #green is channel/debth 1
        b[image==label] = label_colours[label,2] #blue is channel/debth 2
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def storeImageQueue(data, labels, step):
  """ data and labels are all numpy arrays """
  for i in range(BATCH_SIZE):
    index = 0
    im = data[i]
    la = labels[i]
    im = Image.fromarray(np.uint8(im))
    im.save("batch_im_s%d_%d.png"%(step,i))
    writeImage(np.reshape(la,(360,480)), "batch_la_s%d_%d.png"%(step,i))

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_hist(predictions, labels):
  num_class = predictions.shape[3]
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

# def eval_batches(data, sess, eval_prediction=None):
#     """Get all predictions for a dataset by running it in small batches."""
#     size = data.shape[0] # batch_size
#     predictions = np.ndarray(shape=(size, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES), dtype=np.float32)
#     for begin in xrange(0, size, EVAL_BATCH_SIZE):
#       end = begin + EVAL_BATCH_SIZE
#       if end <= size:
#         predictions[begin:end, :] = eval_prediction
#       else:
#         batch_predictions = eval_prediction
#         predictions[begin:, :] = batch_predictions[begin - size:, :]
#     return predictions
