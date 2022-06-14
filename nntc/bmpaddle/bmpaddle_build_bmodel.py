#coding=utf-8

import os, shutil
import bmpaddle

if __name__ == "__main__":
  if os.path.exists('python-output/ch_ppocr_mobile_v2'):
    shutil.rmtree('python-output/ch_ppocr_mobile_v2')
  os.makedirs('python-output/ch_ppocr_mobile_v2')

  bmpaddle.compile(model = 'models/ch_ppocr_mobile_v2.0_cls_infer', # 指定模型目录
                 shapes = [[1,3,32,100]],                           # 输入shapes
                 target = 'BM1684',                                 # 目标设备
                 outdir = 'python-output/ch_ppocr_mobile_v2',       # 指定输出目录
                 net_name = 'ocr-cls',                              # 网络名称
                 cmp = True,                                        # 是否开启比对
                 opt = 2,                                           # 优化级别1,2,3，默认2
                 dyn = False)                                       # 是否动态编译，默认False

