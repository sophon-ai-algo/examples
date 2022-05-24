from cgitb import handler
from json import decoder
import sophon.sail as sail
import cv2
import argparse
import os

def process_padding(handel, ost_name, dst_name, resize_w, resize_h, vpp_flage):
    decoder = sail.Decoder(ost_name)
    input = decoder.read(handel)


    scale_w = float(resize_w) / input.width()
    scale_h = float(resize_h) / input.height()

    temp_resize_w = resize_w
    temp_resize_h = resize_h
    if scale_w < scale_h:
        temp_resize_h = int(input.height()*scale_w)
    else:
        temp_resize_w = int(input.width()*scale_h)
    paddingatt = sail.PaddingAtrr()   
    paddingatt.set_stx(0)
    paddingatt.set_sty(0)
    paddingatt.set_w(temp_resize_w)
    paddingatt.set_h(temp_resize_h)
    paddingatt.set_r(114)
    paddingatt.set_g(114)
    paddingatt.set_b(114)

    bmcv = sail.Bmcv(handel)
    if vpp_flage is 1:
      output_temp = bmcv.vpp_crop_and_resize_padding(
        input,
        0,0,input.width(),input.height(),
        resize_w,resize_h,paddingatt)
    else:
      output_temp = bmcv.crop_and_resize_padding(
        input,
        0,0,input.width(),input.height(),
        resize_w,resize_h,paddingatt)
    bmcv.imwrite(dst_name,output_temp)

def resize_padding_tpu(ost_path, dst_path, resize_w, resize_h):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    file_list = os.listdir(ost_path)
    handel = sail.Handle(0)
    for image_name in file_list:
        ext_name = os.path.splitext(image_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            dst_name = os.path.join(dst_path,"{}_tpu_resize_padding{}".format(image_name[:len(image_name)-len(ext_name)],ext_name))
            ost_name = os.path.join(ost_path,image_name)
            if os.path.exists(dst_name):
                print("Remove: {}".format(dst_name))
                os.remove(dst_name)
            
            process_padding(handel, ost_name, dst_name, resize_w, resize_h, False)

def resize_padding_vpp(ost_path, dst_path, resize_w, resize_h):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    handel = sail.Handle(0)

    file_list = os.listdir(ost_path)
    for image_name in file_list:
        ext_name = os.path.splitext(image_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            dst_name = os.path.join(dst_path,"{}_vpp_resize_padding{}".format(image_name[:len(image_name)-len(ext_name)],ext_name))
            ost_name = os.path.join(ost_path,image_name)
            if os.path.exists(dst_name):
                print("Remove: {}".format(dst_name))
                os.remove(dst_name)
            process_padding(handel, ost_name, dst_name, resize_w, resize_h, True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for resize")
    parser.add_argument('--ost_path', type=str, default="/workspace/test/YOLOX/datasets/ost_data")
    parser.add_argument('--dst_path', type=str, default="/workspace/test/YOLOX/datasets/ost_data_enhance")
    parser.add_argument('--dst_width',type=int, default=640)
    parser.add_argument('--dst_height',type=int, default=640)

    opt = parser.parse_args()
    resize_padding_tpu(opt.ost_path,opt.dst_path,opt.dst_width,opt.dst_height)
    resize_padding_vpp(opt.ost_path,opt.dst_path,opt.dst_width,opt.dst_height)