import os
from PIL import Image
import numpy

"""
    Script that creates result images that puts the label results over input images, with some opacity.

"""

image_path = "../../result_images/input"
label_path = "../../result_images/labels4"
outpath = "../../result_images/combined_images4"

images = sorted(os.listdir(image_path))
labels = sorted(os.listdir(label_path))

for label_name in labels:
    labelnr = label_name.split("testing_image")[1].split(".jpeg")[0]
    label = Image.open(os.path.join(label_path, label_name))
    if(len(labelnr) == 1):
        labelnr = "400"+labelnr
    elif(len(labelnr) == 2):
        labelnr = "40"+labelnr
    elif(len(labelnr) == 4):
        labelnr = "4"+labelnr
    print(labelnr)
    for image_name in images:
        imagenr = image_name.split("map_color_")[1].split(".png")[0]
        print(imagenr)
        if(labelnr == imagenr):
            image = Image.open(os.path.join(image_path, image_name))
            Image.blend(label, image, .7).save(os.path.join(outpath, label_name))
