#!/bin/sh
export BMTFLITE_DIR=$REL_TOP/bmnet/bmtflite

mkdir $BMTFLITE_DIR/build
pushd $BMTFLITE_DIR/build
rm -rf *
cmake ..
make -j8
popd

$BMTFLITE_DIR/build/minimal models/mobilenet_v1_quant_asy.tflite

