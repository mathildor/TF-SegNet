import os
from PIL import Image
import numpy

"""
    Script that converts all pixels in image to correct classes.
    Originally building class was class number 14, so all pixels of type building has value 14.
    It therefor needs to be changed into 1, since the really belong to class 1 and not class 14.
"""

image_path="../../aerial_img_4600/val_images/png/labels"
outpath = "../masterproject/aerial_img_4600/val_images/png/labels_correct"

images = sorted(os.listdir(image_path))

existing_pixelvalues = []

for image_file in images:

    image = Image.open(os.path.join(image_path,image_file))
    pixels = image.load()

    for x in range(0, image.size[0]):
        for y in range(0, image.size[1]):
            if pixels[x,y] not in existing_pixelvalues:
                existing_pixelvalues.append(pixels[x,y])
            if pixels[x,y] == 14:
                pixels[x,y] = 1

    print(existing_pixelvalues)
    break#testing with just one image
    #image.save(os.path.join(outpath, image_file), "PNG", quality=100)
