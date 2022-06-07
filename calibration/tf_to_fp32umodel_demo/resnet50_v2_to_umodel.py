#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
tensorflow model to umodel.
'''
import os
os.environ['GLOG_minloglevel'] = '2'
import ufw.tools as tools

tf_resnet50 = [
    '-m', './models/frozen_resnet_v2_50.pb',
    '-i', 'input',
    '-o', 'resnet_v2_50/predictions/Softmax',
    '-s', '(1, 299, 299, 3)',
    '-d', 'compilation',
    '-n', 'resnet50_v2',
    '-D', './dummy_lmdb',
    '--cmp',
    '--no-transform'
]

if __name__ == '__main__':
    tools.tf_to_umodel(tf_resnet50)
