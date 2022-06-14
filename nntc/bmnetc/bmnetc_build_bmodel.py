#coding=utf-8

import os, shutil
import bmnetc

if __name__ == "__main__":
  if os.path.exists('python-output/det2'):
    shutil.rmtree('python-output/det2')
  os.makedirs('python-output/det2')
  bmnetc.compile(
      model='models/mtcnndet2/det2.prototxt',    # 指定caffe模型结构文件
      weight='models/mtcnndet2/det2.caffemodel', # 指定caffe模型权重文件
      outdir='python-output/det2',               # 指定输出目录
      target='BM1684',                           # 指定目标设备
      opt=2,                                     # 编译优化级别：0,1,2
      dyn=False,                                 # 是否是动态编译
      enable_profile=True,                       # 是否要记录profile信息到bmodel, 默认不记录
      )
