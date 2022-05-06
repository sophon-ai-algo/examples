#!/bin/bash

mkdir -p output/yolov3-tiny
rm -rf output/yolov3-tiny/*
bmnetd --model=models/yolov3-tiny/yolov3-tiny.cfg --weight=models/yolov3-tiny/yolov3-tiny.weights --dyn=false --target=BM1684 --outdir=output/yolov3-tiny --opt=2
