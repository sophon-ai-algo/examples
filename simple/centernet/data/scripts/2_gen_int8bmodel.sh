 #!/bin/bash
 python3 -m ufw.cali.cali_model \
             --net_name 'ctdet_dlav0' \
             --model ../models/ctdet_coco_dlav0_1x.torchscript.pt \
             --cali_image_path ../images/ \
             --cali_image_preprocess='resize_h=512,resize_w=512;mean_value=104.01195:114.03422:119.91659, scale=0.014' \
             --input_shapes '(4,3,512,512)' \
             --test_batch_size 4 \
             --fp32_layer_list '30,33,36' \
             --convert_bmodel_cmd_opt '--max_n=4 '

cp ../models/ctdet_dlav0_batch4/compilation.bmodel ../models/ctdet_coco_dlav0_1output_512_int8_4batch.bmodel
echo "[Success] ../models/ctdet_coco_dlav0_1output_512_int8_4batch.bmodel generated."
