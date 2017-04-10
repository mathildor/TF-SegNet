import os
from PIL import Image


outpath="../masterproject/aerial_img_1400/test_images/jpeg/images/"
path="../masterproject/aerial_img_1400/test_images/images/"


files = sorted(os.listdir(path))

for name in files:
    # outfile = name.replace(' ', '')[:-3]+"jpeg"
    outfile = name.replace(' ', '')[:-3]+"png"
    im = Image.open(os.path.join(path, name))
    im=im.convert('RGB')
    print("Generating png for %s" % name)
    im.thumbnail(im.size)
    #im.save(outfile, "PNG", quality=100)
    # im.save(os.path.join(outpath, outfile), "JPEG", quality=100)
    im.save(os.path.join(outpath, outfile), "PNG", quality=100)
