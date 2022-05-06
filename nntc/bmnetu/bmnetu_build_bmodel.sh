#!/bin/bash

mkdir -p output/mobilenet
rm -rf output/mobilenet/*
bmnetu -model models/mobilenet/mobilenet_deploy_int8_unique_top.prototxt -weight models/mobilenet/mobilenet.int8umodel -max_n 1 -dyn=0  -cmp=1 -target=BM1684 -outdir=output/mobilenet/compilation-mobilenet
