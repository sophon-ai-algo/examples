#!/bin/bash

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
build_dir=$root_dir/build
if [ ! -d "$build_dir" ]; then
  echo "create build dir: $build_dir"
  mkdir -p $build_dir
fi
pushd $build_dir

# export PYTHONPATH=$root_dir/../yolov5
img_size=${1:-640}

function download_pt_model()
{
  # download pt model
  if [ "$img_size" == "1280" ]; then
    if [ ! -f "yolov5s6.pt" ]; then
      echo "download yolov5s6.pt from github"
      wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt
      if [ $? -ne 0 ];then
        echo "failed!"
      fi
    fi
  else
    if [ ! -f "yolov5s.pt" ]; then
      echo "download yolov5s.pt from github"
      wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
      if [ $? -ne 0 ];then
        echo "failed!"
      fi
    fi
  fi

}

function download_val_dataset()
{
  # download val dataset
  if [ ! -f "coco2017val.zip" ]; then
    echo "download coco2017 val dataset from github"
    wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip
    if [ $? -ne 0 ];then
      echo "failed!"
      exit 1
    fi
    echo "unzip coco2017val.zip"
    unzip coco2017val.zip
  fi

}

function download_bmodel()
{
  # TODO: download bmodel
  echo "download bmodel"

}

download_val_dataset

popd
