#!/bin/bash

target_dir=$(dirname $(readlink -f "$0"))

pushd $target_dir
# resnet50.int8.bmodel
mkdir -p ../data/models
./download_from_nas.sh http://219.142.246.77:65000/sharing/KFSrwp8X6
mv resnet50.int8.bmodel ../data/models/


echo "All done!"
popd
