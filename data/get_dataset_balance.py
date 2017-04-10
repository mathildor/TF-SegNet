import os
from PIL import Image
import numpy

""" Calculate median frequency balancing for dataset, to avoid issues with unbalanced dataset. """

image_path="../masterproject/aerial_img_1400/test_images/jpeg/labels"
images = sorted(os.listdir(image_path))

#Saving total number of pixels in dataset belonging to each class
tot_num_class0=0.0
tot_num_class1=0.0
tot_pixels_dataset=0.0

found_pixel_values=[]
class1_freqs = []
class0_freqs = []

for image_file in images:
    #Saving number of pixels in one image belonging to each class
    num_class0 = 0.0 #not building
    num_class1 = 0.0 #building

    image = Image.open(os.path.join(image_path,image_file))
    #image = Image.open('black-white.jpeg')
    pixels = image.load()
    tot_pixels = image.size[0]*image.size[1]

    for x in range(0, image.size[0]):
        for y in range(0, image.size[1]):
            if pixels[x,y] not in found_pixel_values:
                found_pixel_values.append(pixels[x, y])
            if pixels[x,y] == 3:
                num_class0 = num_class0+1
            elif pixels[x,y] == 14:
                num_class1 = num_class1+1

    freq_class0 = num_class0 / tot_pixels
    freq_class1 = num_class1 / tot_pixels

    class0_freqs.append(freq_class0)
    class1_freqs.append(freq_class1)

    tot_num_class0 = tot_num_class0 + num_class0
    tot_num_class1 = tot_num_class1 + num_class1
    tot_pixels_dataset = tot_pixels_dataset + tot_pixels
    print('num_class1 for image: ', image_file)
    print(num_class1)
    print(found_pixel_values)


print('tot_pixels_dataset:')
print(tot_pixels_dataset)
median_freq_class0 = numpy.median(class0_freqs)
median_freq_class1 = numpy.median(class1_freqs)
print('median freq class NOT building is:')
print(median_freq_class0)
print('median freq class building is:')
print(median_freq_class1)

tot_freq_class0= tot_num_class0 / tot_pixels_dataset
tot_freq_class1= tot_num_class1 / tot_pixels_dataset
print('total frequency on whole dataset for NOT buildings are:')
print(tot_freq_class0)
print('total frequency on whole dataset for buildings are:')
print(tot_freq_class1)

class0_score = median_freq_class0 / tot_freq_class0
class1_score = median_freq_class1 / tot_freq_class1

print('Score for class not building: ')
print(class0_score)
print('Score for class building: ')
print(class1_score)
