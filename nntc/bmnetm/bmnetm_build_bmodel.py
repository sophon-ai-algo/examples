#coding=utf-8

import os, shutil
import bmnetm

if __name__ == "__main__":
  if os.path.exists('python-output/lenet'):
    shutil.rmtree('python-output/lenet')
  os.makedirs('python-output/lenet')

  bmnetm.compile(
      model='models/lenet/lenet-symbol.json',  #指定模型结构文件
      weight='models/lenet/lenet-0100.params', #指定模型权重文件
      shapes=[[1, 1, 28, 28]],                 #指定输入shape
      target='BM1684',                         #指定目标设备
      outdir='python-output/lenet',            #指定输出目录, 可选，默认是compilation
      net_name='lenet')                        #设定网络名称，可选，默认是network
