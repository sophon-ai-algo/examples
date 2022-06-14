#coding=utf-8

import os, shutil
import bmnetp

if __name__ == "__main__":
  if os.path.exists('python-output/anchors'):
    shutil.rmtree('python-output/anchors')
  os.makedirs('python-output/anchors')
  bmnetp.compile(model='models/anchors/anchors.pth',  # 模型文件
                 shapes=[[3, 100], [5, 10]],          # 输入shapes
                 target='BM1684',                     # 目标设备
                 net_name='anchors',                  # 网络名称,可选，默认network
                 outdir='python-output/anchors',      # 输出目录,可选，默认compilation
                 cmp=True,                            # 开启比对,可选，默认开启
                 opt=1,                               # 优化等级0,1,2, 可选，默认2
                 dyn=False)                           # 是否动态编译，可选，默认False
