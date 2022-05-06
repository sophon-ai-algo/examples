#!/bin/bash

mkdir -p output/vqvae
rm -rf output/vqvae/*
python3 -m bmnett --model=models/vqvae/vqvae.pb --input_names=Placeholder --shapes=[1,90,180,3] --output_names=valid/forward/ArgMin --net_name=vqvae --target=BM1684 --outdir=output/vqvae --dyn=false
