# !/bin/bash
echo 'yolov5s regression......'
echo 'int8 bmodel'
python3 det_yolov5s.py --bmodel ./yolov5s/compilation.bmodel --imgdir ../../create_lmdb_demo/coco128/images/train2017/ --tpu_id 0 --input ./instances_train2017.json --result ./result.json
echo 'calulate mAP'
python3 cal_mAP.py --anno ./instances_train2017.json --log ./result.json --image-dir ../../create_lmdb_demo/coco128/images/train2017
