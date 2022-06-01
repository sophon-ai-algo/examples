#!/bin/bash

target_dir=$(dirname $(readlink -f "$0"))

pushd $target_dir
# retinaface_mobilenet0.25
mkdir -p ../data/models
./download_from_nas.sh http://219.142.246.77:65000/sharing/qjlkbl7cU
mv retinaface_mobilenet0.25_384x640_fp32_b1.bmodel ../data/models/
./download_from_nas.sh http://219.142.246.77:65000/sharing/eURt5mJS1
mv retinaface_mobilenet0.25_384x640_fp32_b4.bmodel ../data/models/
./download_from_nas.sh http://219.142.246.77:65000/sharing/BGd1IQlXK
mv retinaface_mobilenet0.25.onnx ../data/models/
# station.avi
mkdir -p ../data/videos
./download_from_nas.sh http://219.142.246.77:65000/sharing/LXjz85VVU
mv station.avi ../data/videos
echo "All done!"
popd
