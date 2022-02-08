#!/bin/bash

function convert_imageset_to_lmdb_demo()
{
  rm $IMG_DIR/img_lmdb -rif

  if [ ! -f $IMG_DIR/ImgList.txt ]; then echo "#ERROR: can not find "$IMG_DIR"/ImgList.txt"; return; fi

  convert_imageset --shuffle --resize_height=640 --resize_width=640 \
    $IMG_DIR/  $IMG_DIR/ImgList.txt  $IMG_DIR/img_lmdb
  echo "#INFO: Convert Images to lmdb done"
  return $ret;
}

# IMG_DIR=./images
IMG_DIR=/workspace/YOLOX/data/val2014
convert_imageset_to_lmdb_demo
