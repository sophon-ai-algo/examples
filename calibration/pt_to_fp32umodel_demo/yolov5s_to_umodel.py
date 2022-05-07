#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
pytorch model to umodel.
'''
import os
os.environ['BMNETP_LOG_LEVEL'] = '3'
import ufw.tools as tools

pt_yolov5s = [
    '-m', '../yolov5s_demo/auto_cali_demo/yolov5s_jit.pt',
    '-s', '(1,3,640,640)',
    '-d', 'compilation',
    '--cmp'
]

if __name__ == '__main__':
    tools.pt_to_umodel(pt_yolov5s)
