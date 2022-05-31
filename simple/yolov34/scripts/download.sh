#!/bin/bash
apt update
apt install curl

scripts_dir=$(dirname $(readlink -f "$0"))
echo $scripts_dir

pushd $scripts_dir
# test
mkdir ../data/models
./download_from_nas.sh http://219.142.246.77:65000/sharing/IFj5foNqc
mv yolov4.weights ../data/models/
./download_from_nas.sh http://219.142.246.77:65000/sharing/DNNA06C2W
mv yolov4.cfg ../data/models/
popd
