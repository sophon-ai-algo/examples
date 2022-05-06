#!/bin/bash

mkdir -p output/det2
rm -rf output/det2/*
bmnetc --model=models/mtcnndet2/det2.prototxt --weight=models/mtcnndet2/det2.caffemodel --dyn=false --target=BM1684 --outdir=output/det2 --opt=2
