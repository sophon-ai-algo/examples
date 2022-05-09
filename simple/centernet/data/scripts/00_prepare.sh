#!/bin/bash

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
build_dir=$root_dir/build
if [ ! -d "$build_dir" ]; then
  echo "create build dir: $build_dir"
  mkdir -p $build_dir
fi
pushd $build_dir


function download_val_dataset()
{
  # download val dataset
  if [ ! -f "val2017.zip" ]; then
    # echo "download coco2017 val dataset from cocodataset.org"
    wget http://images.cocodataset.org/zips/val2017.zip
    if [ $? -ne 0 ];then
      echo "failed!"
      exit 1
    fi
  fi
  if [ ! -d "val2017" ]; then
    echo "unzip val2017.zip"
    unzip val2017.zip
  fi
  echo "choose 200 images and copy to ./images"
  # 拷贝200张图片到images文件夹
  ls -l val2017 | sed -n '2,201p' | awk -F " " '{print $9}' | xargs -t -i cp ./val2017/{} ../images
  echo "[Success] 200 jpg files has been located in ../images/"
} 


download_val_dataset

popd
