#!/bin/bash

mkdir -p output/anchors
rm -rf output/anchors/*
python3 -m bmnetp --net_name=anchors --target=BM1684 --opt=1 --cmp=true --shapes="[3,100],[5,10]" --model=models/anchors/anchors.pth --outdir=output/anchors --dyn=false
