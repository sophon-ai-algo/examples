#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
onnx model to umodel.
'''
import os
os.environ['GLOG_minloglevel'] = '3'
import ufw.tools as tools

ppocr_rec = [
    '-m', './models/ppocr_mobile_v2.0_rec',
    '-s', '[(1,3,32,100)]',
    '-i', '["x"]',
    '-o', '["save_infer_model/scale_0.tmp_1"]',
    '-d', 'compilation',
    '--cmp'
]

if __name__ == '__main__':
    tools.pp_to_umodel(ppocr_rec)
