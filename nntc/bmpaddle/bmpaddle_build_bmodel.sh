#!/bin/bash

mkdir -p output/ch_ppocr_mobile_v2.0_cls_infer
rm -rf output/ch_ppocr_mobile_v2.0_cls_infer/*
python3 -m bmpaddle --net_name=ocr-cls --target=BM1684 --opt=2 --cmp=true --shapes="[1,3,32,100]" --model=models/ch_ppocr_mobile_v2.0_cls_infer --outdir=output/ch_ppocr_mobile_v2.0_cls_infer --dyn=false

