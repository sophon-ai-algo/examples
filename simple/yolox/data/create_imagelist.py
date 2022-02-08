import os
import random

IMG_DIR = "/workspace/YOLOX/data/val2014"
max_image_count = 1000

save_list_name = os.path.join(IMG_DIR,"ImgList.txt")
image_list = os.listdir(IMG_DIR)
random.shuffle(image_list)
count_total = 0
str_write = ""
for idx, image_name in enumerate (image_list):
    if image_name[-3:] == "png"  or image_name[-3:] == "jpg" :
        count_total += 1
        line_str = image_name+" 0\n"
        str_write += line_str
    if count_total > max_image_count:
        break

with open(save_list_name,"w+") as fp:
    fp.write(str_write)
    fp.close()


    