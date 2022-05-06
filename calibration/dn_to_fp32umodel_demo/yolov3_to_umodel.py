#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
darknet model to umodel.
'''
import os
import ufw.tools as tools

# Due to the model is too big, we do NOT provide it in this software package.
# You can download yolov3 model form https://pjreddie.com/darknet/.
dn_darknet = [
    '-m', './models/yolov3.cfg',
    '-w', './models/yolov3.weights',
    '-s', '[[1,3,416,416]]',
    '-d', 'compilation'
]

if __name__ == '__main__':
    tools.dn_to_umodel(dn_darknet)
