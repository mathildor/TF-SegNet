import os
from PIL import Image, ImageDraw
# import geojson
import csv

path="../famous_datasets/SpaceNet/processedBuildingLabels/vectordata/geojson"
outpath="../famous_datasets/SpaceNet/processedBuildingLabels/vectordata/png"


files = sorted(os.listdir(path))


out = Image.new("L",(49,87))
dout = ImageDraw.Draw(out)
import csv

for name in files:
    with open(os.path.join(path, name), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dout.point((int(row[0]) / 10,int(row[1]) / 10),fill=int(int(row[2]) * 2.55))
            #print(row[0] + " " + row[1] + " " + row[2])
        out.show()
        break

# for name in files:
#     # outfile = name.replace(' ', '')[:-3]+"jpeg"
#     outfile = name.replace(' ', '')[:-3]+"png"
#     geojson = geojson.loads(name)
#     im = Image.open(os.path.join(path, name))
#     im=im.convert('RGB')
#     print("Generating png for %s" % name)
#     im.thumbnail(im.size)
#     #im.save(outfile, "PNG", quality=100)
#     # im.save(os.path.join(outpath, outfile), "JPEG", quality=100)
#     im.save(os.path.join(outpath, outfile), "PNG", quality=100)
