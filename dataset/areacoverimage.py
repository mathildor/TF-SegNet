from __future__ import division
from __future__ import print_function

import os
import sys
import requests
import shutil
import numpy as np
from PIL import Image
import shutil

from osgeo import gdal
from osgeo import osr
from osgeo import ogr


#Define global variables:
database_name = ""
base_dir  = ""
image_dir = ""
label_dir = ""

wms_url_std = ''
wms_url_hist = ''

#Postgres database specification:
databaseServer = "localhost"
databasePort = "5432"
databaseUser = "postgres"
databasePW = "1234"


""" ------------------- SET PARAMETERS ------------------- """

col_pixel_size = 0.2
col_image_dim = 512
cat_image_dim = 512 #prev 27

#Information about where areas can be found
wms_areas = [
    {
        'name':"Trondheim-2016",
        'rgb_wms_url': wms_url_std,
        'ir_wms_url': wms_url_std
    }
    # ,
    # {
    #     'name':"Oslo-Ostlandet-2016",
    #     'rgb_wms_url': wms_url_hist,
    #     'ir_wms_url': None
    # }
]

datasetName="RGB_Trondheim_full/"
start_img_nr = 0 #Change if creating datasets seperately that should later be merged

""" Chose what to parts of the dataset to create """
#TIF
create_IR_images = False
create_RGB_images = False

convert_IR_to_png = False
convert_RGB_to_png = False

split_IR_dataset = False
split_RGB_dataset = True


""" ------------------------------------------------------- """

# Setup working spatial reference
srs_epsg_nr = 25832
srs = osr.SpatialReference()
srs.ImportFromEPSG(srs_epsg_nr)
srs_epsg_str = 'epsg:{0}'.format(srs_epsg_nr)


feature_table = [
    (0.0, "artype >= 90", "Unknown/novalue"),
    (0.10, "artype =  30 and artreslag =  31", "Barskog"),
    (0.10, "artype =  30 and artreslag =  32", "Loevskog"),
    (0.10, "artype =  30 and artreslag >= 33", "Skog, blandet eller ukjent"),
    (0.10, "artype =  50 and argrunnf >= 43 and argrunnf <= 45", "Jorddekt aapen mark"),
    (0.10, "artype >= 20 and artype < 30", "Dyrket"),
    (0.10, "artype >= 80 and artype < 89", "Water"),
    (0.05, "artype =  50 and argrunnf = 41", "Blokkmark"),
    (0.05, "artype =  50 and argrunnf = 42", "Fjell i dagen"),
    (0.05, "artype =  60", "Myr"),
    (0.01, "artype =  70", "Sne/is/bre"),
    (0.05, "artype =  50 and argrunnf > 45", "Menneskepaavirket eller ukjent aapen mark"),
    (0.20, "artype =  12", "Vei/jernbane/transport"),
    (0.20, "artype >= 10 and artype < 12", "Bebygd"),
    (0.90, "byggtyp is not null", "Bygning"),
    (0.90, "objtype = 'annenbygning' ", "Bygning")
]


def createImageDS(filename, x_min, y_min, x_max, y_max, pixel_size,  srs=None):
    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size) # resolution
    y_res = int((y_max - y_min) / pixel_size) # resolution
    ds = gdal.GetDriverByName('GTiff').Create(filename, x_res,
            y_res, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((
            x_min, pixel_size, 0,
            y_max, 0, -pixel_size,
        ))
    if srs:
        # Make the target raster have the same projection as the source
        ds.SetProjection(srs.ExportToWkt())
    else:
        # Source has no projection (needs GDAL >= 1.7.0 to work)
        ds.SetProjection('LOCAL_CS["arbitrary"]')

    # Set nodata
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    return ds

def create_png_versions_with_correct_pixel_values(path, img_type):
    print("Converting to png for")
    print(img_type)
    save_to_path = base_dir+wms_area_name+"/png/"+ img_type
    create_dir(save_to_path)

    #test_img = Image.open(path+"/map_categories_0.tif")
    #print(test_img)
    
    images = sorted(os.listdir(path))
    print(images)
    print("Images is opened and sorted")
    
    for image_file in images:
        print("For image")
        outfile = image_file.replace(' ', '')[:-3]+"png" #removing .tif ending and replacing with .png
        try:
            im = Image.open(os.path.join(path, image_file))
            print("Generating png for %s" % image_file)
        except Exception:
            print("Could not open image in path:")
            print(os.path.join(path, image_file))
            continue

        if img_type == "labels":
            print("type is labels:")
            pixels = im.load()

            for x in range(0, im.size[0]):
                for y in range(0, im.size[1]):
                    if pixels[x,y] == 14: #converting to correct classname
                        pixels[x,y] = 1
        else:
            im=im.convert('RGB')
        im.thumbnail(im.size)
        im.save(os.path.join(save_to_path, outfile), "PNG", quality=100)


def loadWMS(img_file, url, x_min, y_min, x_max, y_max, x_sz, y_sz, srs, layers, format='image/tiff', styles=None):
    #Styles is set when working with height data
    # Set WMS parameters
        # eksempel url: 'http://www.webatlas.no/wms-orto-std/'
    hdr = None

#http://www.webatlas.no/wms-orto-std/?request=GetMap&layers=Trondheim-2016&width=512&height=512&srs=epsg:25832&format=image/tiff&bbox=560000, 7020000, 570000, 7030000

    params = {
        'request': 'GetMap',
        #'layers': layers,
        'layers': "Trondheim-2016",
        'width': x_sz,
        'height': y_sz,
        'srs': srs,
        'format': format,
        'bbox': '{0}, {1}, {2}, {3}'.format(x_min, y_min, x_max, y_max)
    }
    

    if styles:
        params['styles'] = styles

    # Do request
    for i in range(10):
        try:
            req = requests.get(url, stream=True, params=params, headers=hdr, timeout=None)
            break
        except requests.exceptions.ConnectionError as err:
            time.sleep(10)
    else:
        print("Unable to fetch image")

    # Handle response
    if req.status_code == 200:
        print("request status is 200")
        if req.headers['content-type'] == format:
            # If response is OK and an image, save image file
            with open(img_file, 'wb') as out_file:
                shutil.copyfileobj(req.raw, out_file)
            return True

        else:
            # If no image, print error to stdout
            print("Content-type: ", req.headers['content-type'], " url: ", req.url, " Content: ", req.text, file=sys.stderr)

    # Use existing
    elif req.status_code == 304:
        return True

    # Handle error
    else:
        print("Status: ", req.status_code, " url: ", req.url, file=sys.stderr)

    return False


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def createDataset():
    #Define the global variables as global, to get correct values
    global image_nr
    print("In create dataset")

    cat_pixel_size = col_pixel_size * col_image_dim / cat_image_dim

    connString = "PG: host=%s port=%s dbname=%s user=%s password=%s" % (databaseServer, databasePort, database_name, databaseUser, databasePW)

    conn = ogr.Open(connString)
    layer = conn.GetLayer("ar_bygg") #ar_bygg is the database table

    layer_x_min, layer_x_max, layer_y_min, layer_y_max = layer.GetExtent()

    target_fill = [0] * len(feature_table)

    bbox_size_x = col_pixel_size * col_image_dim
    bbox_size_y = col_pixel_size * cat_image_dim

    #Init boundingbox (bbox)
    x_min = layer_x_min
    y_min = layer_y_min
    x_max = x_min + bbox_size_x
    y_max = y_min + bbox_size_y

    it = 0
    while y_max < (layer_y_max - bbox_size_y): #As long as it hasn't reached end of area
        it = it+1
        if(x_max > layer_x_max + bbox_size_x): #If end of x, skip to next y col
            print("\n Reached x end, moving y length")
            x_min = layer_x_min # reset x_min --> start at layer_x_min again
            #skip one column (y-length):
            y_min = y_min + bbox_size_y
            y_max = y_min + bbox_size_y

        # Create new boundingbox by moving across x-axis
        x_min = x_min + bbox_size_x #Startpunk + lengden av forrige
        x_max = x_min + bbox_size_x

        # Create ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_min, y_min)
        ring.AddPoint(x_max, y_min)
        ring.AddPoint(x_max, y_max)
        ring.AddPoint(x_min, y_max)
        ring.AddPoint(x_min, y_min)

        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # set spatial filter
        layer.SetSpatialFilter(poly)

        # Test for the existence of data
        layer.SetAttributeFilter(None)
        # if no data, go to next
        if layer.GetFeatureCount() < 1:
            #print("Feature count less than 1")
            continue

        good_data = True
        for feature in reversed(feature_table):
            if feature[0] > np.random.random_sample():
                layer.SetAttributeFilter(feature[1])
                #if layer.GetFeatureCount() < 1:
                    #print("Gikk ut paa", feature[2])
                    #good_data = False
                    #break

        if not good_data:
            #print("Not good data")
            continue #skipping to next iteration
        
        # Create image
        target_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(label_dir, "map_categories_{0}.tif".format(image_nr)),
                                                         cat_image_dim, cat_image_dim, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform((
            x_min, cat_pixel_size, 0,
            y_max, 0, -cat_pixel_size,
        ))
        target_ds.SetProjection(srs.ExportToWkt())

        # Fill raster
        no_data = True
        for i, attr_filter in enumerate([feature[1] for feature in feature_table]):
            if attr_filter:
                # Rasterize
                layer.SetAttributeFilter(attr_filter)
                if layer.GetFeatureCount() > 0:
                    no_data = False
                    target_fill[i] += 1
                    if gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[i], options=['ALL_TOUCHED=TRUE']) != 0:
                        raise Exception("error rasterizing layer: %s" % err)

        if no_data:
            print("no data")
            continue #skipping to next iteration

        # Load color image
        colorImgFile = os.path.join(image_dir, "map_color_{0}.tif".format(image_nr))

        #Extracting images
        loadWMS(colorImgFile, wms_url, x_min, y_min, x_max, y_max,
                col_image_dim, col_image_dim, srs_epsg_str, wms_area_name)

        image_nr += 1 #used when setting names for result images

    #print("feature and fill: ")
    #for fill, feature in zip(target_fill, feature_table):
        #print(feature[2], fill)


def split_dataset():
    """ Splitting dataset into three parts: Training, testing, validation"""
    training_size = 0.8
    val_size = 0.1

    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    tot_number = len(images)
    processed_number = 0

    outpath_base = base_dir+"combined_dataset_v2/"
    create_dir(outpath_base)

    for image_name in images:

        imagenr = image_name.split("map_color_")[1].split(".png")[0]
        image = Image.open(os.path.join(image_dir, image_name))

        #Obs! Some images dont have labels version - skip them!
        for label_name in labels:
            labelnr = label_name.split("map_categories_")[1].split(".png")[0]
            if(imagenr == labelnr):
                outpath_images, outpath_labels = set_split_dataset_outpath(processed_number, tot_number, training_size, val_size, outpath_base)
                # os.rename(os.path.join(image_dir, image_name), os.path.join(outpath_images, newName_image) )
                # os.rename(os.path.join(label_dir, label_name), os.path.join(outpath_labels, newName_label) )
                shutil.copy2(os.path.join(image_dir, image_name), os.path.join(outpath_images, image_name) )
                shutil.copy2(os.path.join(label_dir, label_name), os.path.join(outpath_labels, label_name) )

                processed_number = processed_number + 1
                # continue
                break


def set_split_dataset_outpath(processed_number, tot_number, training_size, val_size,outpath_base ):
    if(processed_number > (tot_number * (training_size+val_size)) ):
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

def setAreaVariables(img_type, img_dir_type):
    #For each area:
    #Defining that these variables should be modified globally:
    global database_name, base_dir, image_dir, label_dir

    database_name = wms_area_name.split("-")[0]
    print("DATABASE name is: ")
    print(database_name)
    base_dir  = '/home/mators/aerial_datasets/'+datasetName+img_type+'/'
    create_dir('/home/mators/aerial_datasets')
    create_dir('/home/mators/aerial_datasets/'+datasetName+img_type+'/')
    image_dir = base_dir+wms_area_name+"/"+img_dir_type+'/images/'
    label_dir = base_dir+wms_area_name+"/"+img_dir_type+'/labels/'
    create_dir(image_dir)
    create_dir(label_dir)


if __name__ == "__main__":
    global image_nr , wms_url, wms_area_name

    image_nr=start_img_nr

    for i in range (0, len(wms_areas)):
        #print("Creating images for area:")
        #print(wms_areas[i]["name"])
        if create_RGB_images:
            wms_area_name = wms_areas[i]["name"]
            print("wms are name:")
            print(wms_area_name)
            setAreaVariables("RGB_images", "tif")
            wms_url = wms_areas[i]["rgb_wms_url"]
            createDataset()

        if create_IR_images:
            wms_area_name = wms_areas[i]["name"].split("-")[0]+"-IR-"+wms_areas[i]["name"].split("-")[1]
            setAreaVariables("IR_images", "tif")
            wms_url = wms_areas[i]["ir_wms_url"]
            createDataset()

        if convert_RGB_to_png:
            wms_area_name = wms_areas[i]["name"]
            setAreaVariables("RGB_images", "tif")
            create_png_versions_with_correct_pixel_values(image_dir, "images")
            create_png_versions_with_correct_pixel_values(label_dir, "labels")

        if convert_IR_to_png:
            wms_area_name = wms_areas[i]["name"].split("-")[0]+"-IR-"+wms_areas[i]["name"].split("-")[1]
            #converting to png for images and labels
            print("converting IR images to PNG")
            setAreaVariables("IR_images", "tif")
            create_png_versions_with_correct_pixel_values(image_dir, "images")
            create_png_versions_with_correct_pixel_values(label_dir, "labels")

        if split_RGB_dataset:
            wms_area_name = wms_areas[i]["name"]
            setAreaVariables("RGB_images", "png")
            split_dataset()

        if split_IR_dataset:
            wms_area_name = wms_areas[i]["name"].split("-")[0]+"-IR-"+wms_areas[i]["name"].split("-")[1]
            setAreaVariables("IR_images", "png")
            split_dataset()
