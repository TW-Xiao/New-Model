import os
import cv2
from tqdm import tqdm

img_dir = r"./VOC2007/JPEGImages"
train_txt = r"./VOC2007/ImageSets/Main/trainval.txt"
test_txt = r"./VOC2007/ImageSets/Main/test.txt"

train_store = r"./coco/train2017"
test_store = r"./coco/val2017"

def split_img(img_dir, txt_path, img_type, store_path):
    with open(txt_path, 'r') as F:
        for name in F:
            name = name.strip()
            print(name)

            if "." in img_type:
                img_name = name + img_type
            else:
                img_name = name + "." + img_type

            img = cv2.imread(os.path.join(img_dir, img_name))
            isok = cv2.imwrite(os.path.join(store_path, img_name), img)
            if isok == False:
                raise IOError


# split_img(img_dir=img_dir, txt_path=train_txt, img_type="bmp", store_path=train_store)
split_img(img_dir=img_dir, txt_path=test_txt, img_type="bmp", store_path=test_store)