import os
import sys
import requests
import shutil
import numpy as np
from PIL import Image

""" Splitting dataset into three parts: Training, testing, validation"""

# SET PARAMETERS:
# data_base_path = "../../IR_images/asker_berum/png/"
# data_base_path = "../../IR_images/follo/png/"
data_base_path = "../../IR_images/elverum_vaaler/png/" #use new_number code!!
outpath_base = "../../IR_images/combined_dataset/"

#------------------------

image_path   = data_base_path+"images/"
label_path   = data_base_path+"labels/"

#outpath_images = ""#outpath_base + "train_images/images"
#outpath_labels = ""

training_size = 0.7
val_size = 0.1

images = os.listdir(image_path)
labels = os.listdir(label_path)

tot_number = len(images)
print('tot_number')
print(tot_number)
processed_number = 0

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_outpath(processed_number):
    if(processed_number > (tot_number * (training_size+val_size)) ):
        print("\n TRAINING!")
        outpath_images = outpath_base + "test_images/images"
        outpath_labels = outpath_base + "test_images/labels"
    elif(processed_number > (tot_number * training_size)):
        print("\n Val images, processed_number is:")
        print(processed_number)
        outpath_images = outpath_base + "val_images/images"
        outpath_labels = outpath_base + "val_images/labels"
    else:
        outpath_images = outpath_base + "train_images/images"
        outpath_labels = outpath_base + "train_images/labels"

    create_dir(outpath_images)
    create_dir(outpath_labels)
    return outpath_images, outpath_labels



for image_name in images:

    imagenr = image_name.split("map_color_")[1].split(".png")[0]
    image = Image.open(os.path.join(image_path, image_name))

    #Obs! Some images dont have labels version - skip them!
    for label_name in labels:
        labelnr = label_name.split("map_categories_")[1].split(".png")[0]
        print(labelnr)
        if(imagenr == labelnr):
            outpath_images, outpath_labels = set_outpath(processed_number)

            #new_number =  "10"+image_name.split("map_color_")[1].split(".png")[0] #forgot to set higher number so now multiple images has same number in different parts - have to change that

            newName_image = "map_color_"+new_number+".png"
            newName_label = "map_categories_"+new_number+".png"

            os.rename(os.path.join(image_path, image_name), os.path.join(outpath_images, newName_image) )
            os.rename(os.path.join(label_path, label_name), os.path.join(outpath_labels, newName_label) )
            processed_number = processed_number + 1
            continue
