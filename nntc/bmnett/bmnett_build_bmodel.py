#coding=utf-8

import os, shutil
import bmnett

if __name__ == "__main__":
  if os.path.exists('python-output/vqvae'):
    shutil.rmtree('python-output/vqvae')
  os.makedirs('python-output/vqvae')
  bmnett.compile(model='models/vqvae/vqvae.pb', input_names='Placeholder', shapes=[[1,90,180,3]], output_names='valid/forward/ArgMin', net_name='vqvae', target='BM1684', outdir='python-output/vqvae', dyn=False)

