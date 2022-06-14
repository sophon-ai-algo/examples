#coding=utf-8

import os, shutil
import bmneto

if __name__ == "__main__":
  if os.path.exists('python-output/yolov5s'):
    shutil.rmtree('python-output/yolov5s')
  os.makedirs('python-output/yolov5s')
  bmneto.compile(model = 'models/yolov5s/yolov5s.onnx',  # 指定模型
                 outdir = 'python-output/yolov5s',       # 指定输出目录
                 target = 'BM1684',                      # 指定目标设备
                 shapes = [[1,3,640,640]],               # 指定输入shapes
                 net_name = 'yolov5s',                   # 指定网络名称
                 cmp = True,                             # 比对模式，默认开启
                 opt = 2,                                # 优化等级，0,1,2
                 dyn = False)                            # 动态编译
