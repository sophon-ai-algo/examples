#coding=utf-8

'''
This file is only for demonstrate how to use convert tools to convert
onnx model to umodel.
'''
import os
os.environ['GLOG_minloglevel'] = '3'
import ufw.tools as tools

on_postnet = [
    '-m', './models/postnet.onnx',
    '-s', '[(1, 80, 256)]',
    '-i', '["mel_outputs"]',
    '-d', 'compilation',
    '--cmp'
]

if __name__ == '__main__':
    tools.on_to_umodel(on_postnet)
