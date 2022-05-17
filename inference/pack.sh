#!/usr/bin/env bash

set -e

if [ -z "$1" ];then
  echo "Usage: ./packsh <path-to-model-dirs>"
  exit
fi

MODEL_PATH=$1
COMMON_PARAMS="--config=./cameras.json --output=udp://0.0.0.0:10000"

if [ ! -d "${MODEL_PATH}" ];then
  echo "Directory ${MODEL_PATH} not exist"
fi

function pack_common() {
  rm -rf ${BIN_DIR}
  touch  ${BIN_DIR}
  echo "set -e" >> ${BIN_DIR}
  echo "if [ ! -d \"\$1\" ];then" >> ${BIN_DIR}
  echo "  echo \"Usage: ./run.sh soc | ./run.sh x86\"" >> ${BIN_DIR}
  echo "  exit" >> ${BIN_DIR}
  echo "fi" >> ${BIN_DIR}
  chmod +x $BIN_DIR
}

function pack_cvs10() {
    local BIN_DIR="./release/cvs10/run.sh"
    pack_common cvs10
    echo "./\$1/cvs10 $COMMON_PARAMS --model_type=0 --bmodel=./face_demo.bmodel" >> $BIN_DIR
    cp $MODEL_PATH/face_demo.bmodel ./release/cvs10
    cp ./cvs10/face.jpeg ./release/cvs10
    echo "cvs10 done."
}

function pack_facedetect_demo() {
  local BIN_DIR="./release/facedetect_demo/run.sh"
  pack_common facedetect_demo
  echo "./\$1/facedetect_demo $COMMON_PARAMS --bmodel=./face_demo.bmodel" >> $BIN_DIR
  cp $MODEL_PATH/face_demo.bmodel ./release/facedetect_demo
  echo "facedetect_demo done."
}

function pack_openpose_demo() {
  local BIN_DIR="./release/openpose_demo/run.sh"
  pack_common openpose_demo
  echo "./\$1/openpose_demo $COMMON_PARAMS --bmodel=./openpose_coco_17_216_384.bmodel" >> $BIN_DIR
  cp $MODEL_PATH/openpose_coco_17_216_384.bmodel ./release/openpose_demo
  cp $MODEL_PATH/openpose_body_25_216_384.bmodel ./release/openpose_demo
  echo "openpose_demo done."
}

function pack_retinaface_demo() {
  local BIN_DIR="./release/retinaface_demo/run.sh"
  pack_common retinaface_demo
  echo "./\$1/retinaface_demo $COMMON_PARAMS --max_batch=4 --bmodel=./retinaface_mobilenet0.25_384x640_fp32_b4.bmodel" >> $BIN_DIR
  cp $MODEL_PATH/retinaface_mobilenet0.25_384x640_fp32_b4.bmodel ./release/retinaface_demo
  echo "retinaface_demo done."
}

function pack_video_stitch_demo() {
  local BIN_DIR="./release/video_stitch_demo/run.sh"
  pack_common video_stitch_demo
  echo "./\$1/video_stitch_demo $COMMON_PARAMS --max_batch=4 --bmodel=./yolov5s_4b_int8_v21.bmodel --num=4 --skip=2 " >> $BIN_DIR
  cp $MODEL_PATH/yolov5s_4b_int8_v21.bmodel ./release/video_stitch_demo
  echo "video_stitch_demo done."
}

function pack_yolov5s_demo() {
  local BIN_DIR="./release/yolov5s_demo/run.sh"
  pack_common yolov5s_demo
  echo "./\$1/yolov5s_demo $COMMON_PARAMS --max_batch=4 --bmodel=./yolov5s_4b_int8_v21.bmodel" >> $BIN_DIR
  cp $MODEL_PATH/yolov5s_4b_int8_v21.bmodel ./release/yolov5s_demo
  echo "yolov5s_demo done."
}

function pack_safe_hat_detect_demo() {
    local BIN_DIR="./release/safe_hat_detect_demo/run.sh"
    pack_common hat_detect_demo
    echo "./\$1/safe_hat_detect_demo $COMMON_PARAMS --bmodel=./person_detect_safehat_v2.bmodel" >> $BIN_DIR
    cp $MODEL_PATH/person_detect_safehat_v2.bmodel ./release/safe_hat_detect_demo
    echo "safe_hat_detect_demo done."
}

function pack_multi_demo() {
    local BIN_DIR="./release/multi_demo/run.sh"
    pack_common multi_demo
    echo "./\$1/multi_demo --config=./cameras_v1.json" >> $BIN_DIR
    cp $MODEL_PATH/yolov5s_4b_int8_v21.bmodel ./release/multi_demo
    cp $MODEL_PATH/yolov5s.bmodel ./release/multi_demo
    echo "multi_demo done."
}

pack_cvs10
pack_facedetect_demo
pack_openpose_demo
pack_retinaface_demo
pack_video_stitch_demo
pack_yolov5s_demo
pack_safe_hat_detect_demo
pack_multi_demo
cp /data/others/* release/
cp release.md release/README.md
cp /data/workspace/media/100new.264 release/
echo "Done."


