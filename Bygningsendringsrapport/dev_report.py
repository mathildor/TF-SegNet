
# Read label and create buffer

import os
from PIL import Image
from PIL import ImageChops
import numpy
numpy.set_printoptions(threshold=numpy.nan)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import rasterio #To vectorize a raster
# import gdal

label = "C:\\Users\mators\\Documents\\Master-data\\test-set\\label.png" #Original true label
adjusted_label = "C:\\Users\mators\\Documents\\Master-data\\test-set\\edited_label_removed_building.png" #One building removed to test if it can be detected
net = "C:\\Users\\mators\\Documents\\Master-data\\test-set\\net.png" #label returned by neural net
tiff_img = "C:\\Users\\mators\\Documents\\Master-data\\test-set\\tiff_img.tif" #tiff input image containing geographical data 

input_ir = "C:\\Users\\mators\\Documents\\Master-data\\test-set\\input.png"
outpath = "C:\\Users\\mators\\Documents\\Master-data\\test-set\\"

label = Image.open(label)
net_label = Image.open(net)
adjusted_label = Image.open(adjusted_label)

def toRaster(img):
    print("Convert to raster")

def toVector(img):
    print("Convert to vector")

def visualize_img(img):
    img.show()

#Set buildings to white color to make them visible
def set_building_color(image):
    pixels = image.load()
    if(image.mode == "RGB"): #Extracting one channel to create gray-scale
        red, green, blue = image.split()
        #Extracting red channel since I know the net labels have some red color for the building pixels
        image = red
        pixels = image.load()
    if(image.mode == "RGBA"): #Extracting one channel to create gray-scale
        red, green, blue, alpha = image.split()
        #Extracting red channel since I know the net labels have red color for the building pixels
        image = red
        pixels = image.load()
    for x in range(0, image.size[0]):
        for y in range(0, image.size[1]):
            if pixels[x,y] >= 1:
                pixels[x,y] = 255
    #image.save(outpath+"edited_label.png")
    return image

def calculate_difference(img1, img2):
    pixels1 = img1.load()
    pixels2 = img2.load()
    for x in range(0, img1.size[0]):
        for y in range(0, img1.size[1]):
            if (pixels1[x,y] - pixels2[x,y]) < 0:
                pixels1[x,y] = 40
            else:
                pixels1[x,y] = pixels1[x,y] - pixels2[x,y]
    return img1


#Find a more effective way to search later
def get_dev_area_pixels(rest_net_label_pos, building_pos_list, dist_limit):
    # Find distance inbetween inserted pixel position and a building in img
    # pixel_pos is for example (3,4)
    dev_area_pixels = []
    dev_array = numpy.zeros(net_label.size)
    for label_pos in rest_net_label_pos:
        least_diff = 1000000
        for building_pix in building_pos_list:
            distance = numpy.sqrt( ((label_pos[0] - building_pix[0])**2) + ((label_pos[1] - building_pix[1])**2) )
            if distance < least_diff:
                least_diff = distance
            if least_diff <= dist_limit: #Do not need to search further when first close building is discovered
                break
        if least_diff > dist_limit:
            dev_area_pixels.append(label_pos)
    return dev_area_pixels

def create_dev_area_image(img, positions):
    #Have to use one of the original images and edit from them because of saved metadata
    pixels = img.load()
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
            if (x,y) in positions:
                pixels[x,y] = 255
            else:
                pixels[x,y] = 0
    img.save(outpath+"dev_area.png", "PNG")
    return img
    
def get_building_positions(img):
    pos_list = []
    pixels = img.load()
    for x in range(0, img.size[0]):
        for y in range(0, img.size[1]):
            #print(pixels[x,y][0])
            if (pixels[x,y] == 255):
                pos_list.append((x,y))
    return pos_list

def convert_to_tiff(tiff_img, png_img):
    tiff = Image.open(tiff_img)
    print("TIFF size:")
    print(tiff.size)
    png_pixels = png_img.load()
    red, green, blue = tiff.split()
    tiff = red
    tiff_pixels = tiff.load()
    for x in range(0, tiff.size[0]):
        for y in range(0, tiff.size[1]):
            print("Tiff pixels are:")
            print(tiff_pixels[x,y])
            tiff_pixels[x,y] = png_pixels[x,y]
    return tiff


net_label=set_building_color(net_label)
adjusted_label = set_building_color(adjusted_label)

#Saving new image where all pixels in net_label that corresponds to the true label is removed
rest_net_label = calculate_difference(net_label, adjusted_label)
#building_pos_list = get_building_positions(adjusted_label)
#net_building_pos_list = get_building_positions(rest_net_label)
#dev_area = get_dev_area_pixels(net_building_pos_list, building_pos_list, 10)
#dev_img = create_dev_area_image(rest_net_label, dev_area)

#dev_tiff_img = convert_to_tiff(tiff_img, dev_img)

dev_tiff_img = convert_to_tiff(tiff_img, set_building_color(adjusted_label))
dev_tiff_img.save(outpath+"dev_area.tiff", "TIFF")
dev_tiff_img.save()
visualize_img(dev_tiff_img)
#visualize_img(calculate_difference(set_building_color(net_label), set_building_color(adjusted_label)))
#visualize_img(set_building_color(net_label))
#visualize_img(calculate_difference(set_building_color(net_label), set_building_color(label)))