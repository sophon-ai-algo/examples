#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
pytorch model to umodel.
'''
import os
import ufw.tools as tools

pt_mobilenet = [
    '-m', '/workspace/YOLOX/models/yolox_m.pt',
    '-s', '(1,3,640,640)',
    '-d', '/workspace/YOLOX/models/yolox_m',
    '-D', '/workspace/YOLOX/data/val2014/img_lmdb',
    '--cmp'
]

if __name__ == '__main__':
    tools.pt_to_umodel(pt_mobilenet)