#coding=utf-8

import os, shutil
import bmnetp

if __name__ == "__main__":
  if os.path.exists('python-output/anchors'):
    shutil.rmtree('python-output/anchors')
  os.makedirs('python-output/anchors')
  bmnetp.compile(model='models/anchors/anchors.pth', shapes=[[3,100],[5,10]], net_name='anchors', target='BM1684', outdir='python-output/anchors', cmp=True, opt=1, dyn=False)
