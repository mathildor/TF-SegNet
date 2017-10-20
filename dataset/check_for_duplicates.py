
import os
from PIL import Image
import numpy

from PIL import ImageChops

""" TESTED:

    No duplicates in:
        - within validation images first part (stopped because of training - took to much time)

 """

image_path="../../IR_images/combined_dataset/val_images/images"
# image_path="../../IR_images/combined_dataset/val_images/images"

images = sorted(os.listdir(image_path))


for image_file_1 in images:
    for image_file_2 in images:
        image1 = Image.open(os.path.join(image_path,image_file_1))
        image2 = Image.open(os.path.join(image_path,image_file_2))
        #pixels = image.load()

        if ImageChops.difference(image1, image2).getbbox() is None:
        # if(image1==image2):# and image_file_1 != image_file_2):
            print("Same image!!!")
            print(image_file_1)
            print(image_file_2)
        # else:
        #     print("not same")
        #     print(image_file_1)
        #     print(image_file_2)
