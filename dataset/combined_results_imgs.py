import os
from PIL import Image
import numpy

"""
    Script that creates result images that puts the label results over input images, with some opacity.

"""

# image_path = "../../IR_images/combined_dataset/"
# label_path = "../result_imgs/for_combined_images/"
# image_path = "../../aerial_datasets/IR_RGB_0.1res/IR_images/combined_dataset/test_images/images" #Input images
image_path = "../../aerial_datasets/IR_RGB_0.1res/IR_images/combined_dataset/test_images/images/" #Ground truth
label_path = "../result_imgs_IR/"
outpath = "../result_imgs_combined/"

images = sorted(os.listdir(image_path))
labels = sorted(os.listdir(label_path))

#IR test = 309
#IR labels = 309

#First rename all image-labels based on what number in sorted list they are? Smallest number = 0, next smallest number = 1 and so on?

processed = 0

for image_name in images:
    for label_name in labels:
        labelnr = label_name.split("testing_image")[1].split(".png")[0]

        if(int(labelnr) == int(processed)):
            label = Image.open(os.path.join(label_path, label_name))
            print("same number!")
            image = Image.open(os.path.join(image_path, image_name))
            # image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
            print(label)
            print(image)
            blend_img = Image.blend(label, image, .7)
            blend_img.save(os.path.join(outpath, image_name))
            # Image.blend(label, image, .7).save(os.path.join(outpath, label_name+"_label"))
    processed +=1


# for label_name in labels:
#     labelnr = label_name.split("testing_image")[1].split(".png")[0]
#     label = Image.open(os.path.join(label_path, label_name))
#     if(len(labelnr) != -1):
#         # labelnr = "400"+labelnr
#         # print(int(float(labelnr)))
#         labelnr = int(float(labelnr)) + 4271
#         print("new label number is: ")
#         print(labelnr)
#     # elif(len(labelnr) == 2):
#     #     labelnr = "40"+labelnr
#     # elif(len(labelnr) == 4):
#     #     labelnr = "4"+labelnr
#     # print(labelnr)
#     for image_name in images:
#         imagenr = image_name.split("map_color_")[1].split(".png")[0]
#         print('imagenr is:')
#         print(imagenr)
#         if(int(labelnr) == int(imagenr)):
#             print("same number!")
#             image = Image.open(os.path.join(image_path, image_name))
#             Image.blend(label, image, .7).save(os.path.join(outpath, label_name))
