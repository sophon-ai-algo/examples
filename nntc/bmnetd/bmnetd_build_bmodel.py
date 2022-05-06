#coding=utf-8

import os, shutil
import bmnetd

if __name__ == "__main__":
  if os.path.exists('python-output/yolov3-tiny'):
    shutil.rmtree('python-output/yolov3-tiny')
  os.makedirs('python-output/yolov3-tiny')
  bmnetd.compile(model='models/yolov3-tiny/yolov3-tiny.cfg', weight='models/yolov3-tiny/yolov3-tiny.weights', outdir='python-output/yolov3-tiny', target='BM1684', opt=2, dyn=False)
