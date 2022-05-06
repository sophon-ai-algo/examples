#coding=utf-8

import os, shutil
import bmpaddle

if __name__ == "__main__":
  if os.path.exists('python-output/ch_ppocr_mobile_v2.0_cls_infer'):
    shutil.rmtree('python-output/ch_ppocr_mobile_v2.0_cls_infer')
  os.makedirs('python-output/ch_ppocr_mobile_v2.0_cls_infer')
  bmpaddle.compile(model = 'models/ch_ppocr_mobile_v2.0_cls_infer', 
                 outdir = 'python-output/ch_ppocr_mobile_v2.0_cls_infer', 
                 target = 'BM1684', 
                 shapes = [[1,3,32,100]], 
                 net_name = 'ocr-cls', 
                 cmp = True, 
                 opt = 2, 
                 dyn = False)

