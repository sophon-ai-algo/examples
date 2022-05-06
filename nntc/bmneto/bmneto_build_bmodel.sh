#!/bin/bash

mkdir -p output/yolov5s
rm -rf output/yolov5s/*
python3 -m bmneto --net_name=yolov5s --target=BM1684 --opt=2 --cmp=true --shapes="[1,3,640,640]" --model=models/yolov5s/yolov5s.onnx --outdir=output/yolov5s --dyn=false
