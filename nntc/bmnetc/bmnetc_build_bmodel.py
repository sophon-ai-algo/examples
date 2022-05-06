#coding=utf-8

import os, shutil
import bmnetc

if __name__ == "__main__":
  if os.path.exists('python-output/det2'):
    shutil.rmtree('python-output/det2')
  os.makedirs('python-output/det2')
  bmnetc.compile(model='models/mtcnndet2/det2.prototxt', weight='models/mtcnndet2/det2.caffemodel', outdir='python-output/det2', target='BM1684', opt=2, dyn=False)
