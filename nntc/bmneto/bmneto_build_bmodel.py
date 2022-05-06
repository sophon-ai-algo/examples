#coding=utf-8

import os, shutil
import bmneto

if __name__ == "__main__":
  if os.path.exists('python-output/yolov5s'):
    shutil.rmtree('python-output/yolov5s')
  os.makedirs('python-output/yolov5s')
  bmneto.compile(model = 'models/yolov5s/yolov5s.onnx', 
                 outdir = 'python-output/yolov5s', 
                 target = 'BM1684', 
                 shapes = [[1,3,640,640]], 
                 net_name = 'yolov5s', 
                 cmp = True, 
                 opt = 2, 
                 dyn = False)
