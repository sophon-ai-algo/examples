#coding=utf-8

import os, shutil
import bmnetm

if __name__ == "__main__":
  if os.path.exists('python-output/lenet'):
    shutil.rmtree('python-output/lenet')
  os.makedirs('python-output/lenet')
  bmnetm.compile(model='models/lenet/lenet-symbol.json', weight='models/lenet/lenet-0100.params', shapes=[[1,1,28,28]], target='BM1684', outdir='python-output/lenet', net_name='lenet')
