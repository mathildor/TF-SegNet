import os
import sys

import numpy as np
from PIL import Image


image_path = "../../IR_images/combined_dataset/test_images/images/map_color_4271.png"

img = Image.open(image_path)

print(img.mode)
print(img.size)
