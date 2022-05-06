#!/bin/bash

mkdir -p output/lenet
rm -rf output/lenet/*
python3 -m bmnetm --net_name=lenet --model=models/lenet/lenet-symbol.json --weight=models/lenet/lenet-0100.params --shapes=[1,1,28,28] --target=BM1684 --outdir=output/lenet
