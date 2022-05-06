#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
caffemodel to umodel.
'''
import os
os.environ['GLOG_minloglevel'] = '2'
import ufw.tools as tools

cf_resnet50 = [
    '-m', './models/ResNet-50-test.prototxt',
    '-w', './models/ResNet-50-model.caffemodel',
    '-s', '(1,3,224,224)',
    '-d', 'compilation',
    '--cmp'
]

if __name__ == '__main__':
    tools.cf_to_umodel(cf_resnet50)
