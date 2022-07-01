python3 -m ufw.cali.cali_model \
    --net_name 'yolov5s' \
    --model ./yolov5s_jit.pt \
    --cali_image_path ../../create_lmdb_demo/coco128/images/train2017/ \
    --cali_image_preprocess 'resize_h=640,resize_w=640;scale=0.003921569,bgr2rgb=True' \
    --input_shapes '(1,3,640,640)' \
    --postprocess_and_calc_score_class=feature_similarity \
    --try_cali_accuracy_opt='-fpfwd_outputs=< 24 >14,< 24 >51,< 24 >82'