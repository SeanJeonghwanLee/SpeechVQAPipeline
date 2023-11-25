import os
from PIL import Image
from glob import glob

root_path = '/home/seanlee/class/deeplearning/image/train/*'
image_list = glob(root_path)

for image in image_list[:100]:
    
    img = Image.open(image)
    print(img.size)