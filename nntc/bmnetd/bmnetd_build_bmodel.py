#coding=utf-8

import os, shutil
import bmnetd

if __name__ == "__main__":
  if os.path.exists('python-output/yolov3-tiny'):
    shutil.rmtree('python-output/yolov3-tiny')
  os.makedirs('python-output/yolov3-tiny')
  bmnetd.compile(
      model='models/yolov3-tiny/yolov3-tiny.cfg',       # 指定网络结构文件
      weight='models/yolov3-tiny/yolov3-tiny.weights',  # 指定网络权重
      outdir='python-output/yolov3-tiny',               # 输出目录
      target='BM1684',                                  # 目标设备
      opt=2,                                            # 优化等级，可以为0,1,2
      dyn=False)                                        # 是否是动态编译
