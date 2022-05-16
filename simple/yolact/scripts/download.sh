#!/bin/bash
apt update
apt install curl

scripts_dir=$(dirname $(readlink -f "$0"))
echo $scripts_dir

pushd $scripts_dir
# test
mkdir ../data/models
./download_from_nas.sh http://219.142.246.77:65000/sharing/J2vbKoPDY
mv yolact_base_54_800000.pth ../data/models/
popd
