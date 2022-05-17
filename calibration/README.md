# calibration examples

本examples包含如下子模块：

- auto_cali_demo, auto_cali的demo
- yolov5s_demo，使用auto_cali量化yolov5s的demo
- caffemodel_to_fp32model_demo, 使用ufw将 caffe 编译为fp32umodel的demo
- dn_to_fp32umodel_demo, 使用ufw将 darknet 编译为fp32umodel的demo
- mx_to_fp32umodel_demo, 使用ufw将 mxnet 编译为fp32umodel的demo
- pp_to_fp32umodel_demo, 使用ufw将 paddlepaddle 编译为fp32umodel的demo
- pt_to_fp32umodel_demo, 使用ufw将 pytorch 编译为fp32umodel的demo
- tf_to_fp32umodel_demo, 使用ufw将 tensorflow 编译为fp32umodel的demo
- create_lmdb_demo, 使用ufwio制作lmdb数据集的demo
- classify_demo, 
- face_demo, 人脸检测网络量化，量化网络后可以把图片上人脸框出
- object_detection_python_demo, python物体检测demo

## 准备工作
请先运行`bash ./prepare.sh` 下载examples所需模型和数据集

