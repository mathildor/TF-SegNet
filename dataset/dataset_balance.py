import os
from PIL import Image
import numpy

""" Calculate median frequency balancing for dataset, to avoid issues with unbalanced dataset.

    we weight each pixel by
        ac = median_freq / freq(c)
    where freq(c) is the number of pixels of class c divided by the total number of pixels in images where c is present,
    and median_freq is the median of these frequencies.

"""

# image_path="../masterproject/aerial_img_1400/test_images/jpeg/labels"
image_path="../../aerial_datasets/IR_RGB_0.1res/IR_images/combined_dataset/train_images/labels/"
images = sorted(os.listdir(image_path))

#Saving total number of pixels in dataset belonging to each class
tot_num_class0=0.0
tot_num_class1=0.0
tot_pixels_dataset=0.0

class1_freqs = []
class0_freqs = []
class_freqs = []

for image_file in images:
    #Saving number of pixels in one image belonging to each class
    print("inspecting image %s"%image_file)
    num_class0 = 0.0 #not building
    num_class1 = 0.0 #building

    image = Image.open(os.path.join(image_path,image_file))
    pixels = image.load()

    img_array = numpy.asarray(image, dtype="float")

    num_class0 = float(numpy.count_nonzero(img_array == 0.0))
    num_class1 = float(numpy.count_nonzero(img_array == 1.0))
    tot_pixels = num_class0 + num_class1

    freq_class0 = num_class0 / tot_pixels
    freq_class1 = num_class1 / tot_pixels

    class_freqs.append(freq_class0)
    class_freqs.append(freq_class1)

    tot_num_class0 = tot_num_class0 + num_class0
    tot_num_class1 = tot_num_class1 + num_class1
    tot_pixels_dataset = tot_pixels_dataset + tot_pixels

median_freq = numpy.median(class_freqs)

tot_freq_class0= tot_num_class0 / tot_pixels_dataset
tot_freq_class1= tot_num_class1 / tot_pixels_dataset

class0_score = median_freq / tot_freq_class0
class1_score = median_freq / tot_freq_class1

print('Score for class not building: ')
print(class0_score)
print('Score for class building: ')
print(class1_score)

""" RESULT:
Score for class not building:
0.545399958039
Score for class building:
6.006613019
"""
