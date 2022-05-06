#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
mxnet model to umodel.
'''
import os
os.environ['GLOG_minloglevel'] = '3'
import ufw.tools as tools

mx_mobilenet = [
    '-m', './models/mobilenet0.25-symbol.json',
    '-w', './models/mobilenet0.25-0000.params',
    '-s', '(1,3,128,128)',
    '-d', 'compilation',
    '-D', '../classify_demo/lmdb/imagenet_s/ilsvrc12_val_lmdb_with_preprocess',
    '--cmp'
]

if __name__ == '__main__':
    tools.mx_to_umodel(mx_mobilenet)
