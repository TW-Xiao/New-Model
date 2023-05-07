import os
import shutil
import cv2 as cv
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

folder_dir = '/home/icisee/XTW/声学/pytorch_simple_CenterNet_45-master/data/coco'
txt_dir = 'val2017'

#### 挑出1000张原图片
data_folder = os.path.join(folder_dir, txt_dir)  # 文件路径
for v in os.listdir(data_folder):
    v = os.path.splitext(v)[0]
    # 图片路径，相对路径
    image_path = r'/home/icisee/XTW/声学/pytorch_simple_CenterNet_45-master/data/VOCdevkit/VOC2007/JPEGImages_break'
    image_name = v
    image_path1 = r'/home/icisee/XTW/声学/pytorch_simple_CenterNet_45-master/data/VOCdevkit/VOC2007/val2017_break'
    print(v)
    image_path_name = os.path.join(image_path, image_name + ".bmp")
    image_path_name1 = os.path.join(image_path1, image_name + ".bmp")
    print(image_path_name)
    # 读取图片
    image = Image.open(image_path_name)
    # 保存图片
    image.save(image_path_name1)