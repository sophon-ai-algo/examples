#coding=utf-8

import os, shutil
import bmnett

if __name__ == "__main__":
  if os.path.exists('python-output/vqvae'):
    shutil.rmtree('python-output/vqvae')
  os.makedirs('python-output/vqvae')

  bmnett.compile(model='models/vqvae/vqvae.pb',       # 模型位置
                 input_names='Placeholder',           # 输入名称, 多个用','分隔
                 shapes=[[1, 90, 180, 3]],            # 输入shapes
                 target='BM1684',                     # 目标设备
                 output_names='valid/forward/ArgMin', # 输出名称，可选，默认全部悬空的tensor作为输出
                 outdir='python-output/vqvae',        # 输出目录
                 net_name='vqvae',                    # 网络名称
                 dyn=False)                           # 是否动态编译

