from __future__ import division
from __future__ import print_function

import os
import sys
import requests
import shutil
import numpy as np
from PIL import Image

from osgeo import gdal
from osgeo import osr
from osgeo import ogr


# Setup working spatial reference
srs_epsg_nr = 25832
srs = osr.SpatialReference()
srs.ImportFromEPSG(srs_epsg_nr)
srs_epsg_str = 'epsg:{0}'.format(srs_epsg_nr)


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

def create_png_versions(path, img_type):
    save_to_path = base_dir+"png/"+ img_type
    #create png directory and type directory
    create_dir(base_dir+"png")
    create_dir(save_to_path)

    images = sorted(os.listdir(path))
    for image in images:
        outfile = image.replace(' ', '')[:-3]+"png" #removing .tif ending and replacing with .png
        im = Image.open(os.path.join(path, image))
        print("Generating png for %s" % image)
        im=im.convert('RGB')
        im.thumbnail(im.size)
        im.save(os.path.join(save_to_path, outfile), "PNG", quality=100)


def loadWMS(img_file, url, x_min, y_min, x_max, y_max, x_sz, y_sz, srs, layers, format='image/tiff', styles=None ):
    # Set WMS parameters
    #url: 'http://www.webatlas.no/wms-orto-std/'
    hdr = None
    params = {
        'request': 'GetMap',
        'layers': layers,
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
            print(err)
            time.sleep(10)
    else:
        print("Unable to fetch image from: " + url + " with parameters: " + params)

    # Handle response
    if req.status_code == 200:
        if req.headers['content-type'] == format:
            # If response is OK and an image, save image file
            with open(img_file, 'wb') as out_file:
                shutil.copyfileobj(req.raw, out_file)
                #save_as_png(img_file)

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
    (0.90, "byggtyp is not null",  "Bygning")
]

databaseServer = "localhost"
databasePort = "5432"
databaseName = "map_files"
databaseUser = "postgres"
databasePW = "motgikjo"

wms_url = 'http://www.webatlas.no/wms-orto-std/'
#image_dir = 'C:/Users/runaas/data/Klassifiseringstest/testimg'
# image_dir = './aerial_img/test_images'
base_dir  = './aerial_img_val/'
image_dir = base_dir+'images/'
label_dir = base_dir+'labels/'

#create directory structure
create_dir(base_dir)
create_dir(image_dir)
create_dir(label_dir)

col_pixel_size = 0.2
col_image_dim = 512
cat_image_dim = 512 #prev 27
cat_pixel_size = col_pixel_size * col_image_dim / cat_image_dim
max_image_nr = 1400
image_nr = 1201

connString = "PG: host=%s port=%s dbname=%s user=%s password=%s" % (databaseServer, databasePort, databaseName, databaseUser, databasePW)

conn = ogr.Open(connString)
layer = conn.GetLayer( "ar_bygg" )

layer_x_min, layer_x_max, layer_y_min, layer_y_max = layer.GetExtent()

target_fill = [0] * len(feature_table)

while image_nr <= max_image_nr:
    # Create boundingbox
    #np.random.random() Return random floats in the half-open interval [0.0, 1.0).
    x_min = np.random.random_sample() * (layer_x_max - layer_x_min) + layer_x_min
    y_min = np.random.random_sample() * (layer_y_max - layer_y_min) + layer_y_min
    x_max = x_min + col_pixel_size * col_image_dim
    y_max = y_min + col_pixel_size * col_image_dim

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
        continue

    good_data = True
    for feature in reversed(feature_table):
        if feature[0] > np.random.random_sample():
            layer.SetAttributeFilter(feature[1])
            if layer.GetFeatureCount() < 1:
                print("Gikk ut paa", feature[2])
                good_data = False
                break

    if not good_data:
        continue

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
        continue

    # Load color image
    colorImgFile = os.path.join(image_dir, "map_color_{0}.tif".format(image_nr))
    loadWMS(colorImgFile, wms_url, x_min, y_min, x_max, y_max,
            col_image_dim, col_image_dim, srs_epsg_str, 'ortofoto')

    # Load elevation image
    #elevationImageFile = os.path.join(image_dir, "map_elevation_{0}.tif".format(image_nr))
    #loadWMS(elevationImageFile, wms_url, x_min, y_min, x_max, y_max,
    #        col_image_dim, col_image_dim, srs_epsg_str, 'hoyderaster', styles='raw')

    image_nr += 1

print("feature and fill: ")
for fill, feature in zip(target_fill, feature_table):
    print(feature[2], fill)

#converting to jpeg for images and labels
create_png_versions(image_dir, "images")
create_png_versions(label_dir, "labels")
