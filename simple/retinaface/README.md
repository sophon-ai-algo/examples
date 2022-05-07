#Retinaface Face Detect Demos
## 1. Introduction
data目录，包含了测试图片及模型

python目录, 使用opencv或bmcv做前处理，sail作推理，numpy做后处理

cpp目录，cpp的示例程序

>**注意：** cpp和python示例程序所用的模型结构不同  
> cpp代码对应模型：face_detection_fp32_b1/b4.bmodel，对应9个输出tensor；   
> python代码对应模型：retinaface_***_384x640_fp32_b1/b4.bmodel，对应3个输出tensor。若要使用backbone为resnet50的模型，请通过以下nas网盘链接下载：http://219.142.246.77:65000/sharing/m6cELCAx7


## 2. python demo usage
Python代码无需编译，无论是x86 SC5平台还是arm SE5平台配置好环境之后就可直接运行。

### 2.1  环境安装配置

``` shell
$ cd python
$ pip3 install -r requirements.txt
```

### 2.2  测试命令
#### 2.2.1 查看测试命令参数
``` shell
# 使用opencv做前处理
$ python3 retinaface_sophon_opencv.py --help
# 使用bmcv做前处理
$ python3 retinaface_sophon_bmcv.py --help
```

#### 2.2.2  测试图片

``` shell
# 使用opencv做前处理
$ python3 retinaface_sophon_opencv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/images/face1.jpg --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
# 使用bmcv做前处理
$ python3 retinaface_sophon_bmcv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/images/face1.jpg --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
```
测试结束后会将预测图片保存至result_imgs目录下，并打印相关测试时间如下：
``` shell
+--------------------------------------------------------------------------------+
|                           Running Time Cost Summary                            |
+------------------------+----------+--------------+--------------+--------------+
|        函数名称        | 运行次数 | 平均耗时(秒) | 最大耗时(秒) | 最小耗时(秒) |
+------------------------+----------+--------------+--------------+--------------+
|     predict_numpy      |    1     |    0.082     |    0.082     |    0.082     |
| preprocess_with_opencv |    1     |    0.007     |    0.007     |    0.007     |
|      infer_numpy       |    1     |    0.008     |    0.008     |    0.008     |
|      postprocess       |    1     |    0.029     |    0.029     |    0.029     |
+------------------------+----------+--------------+--------------+--------------+
```
#### 2.2.2  测试视频
``` shell
# 使用opencv做前处理
$ python3 retinaface_sophon_opencv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/images/dance.mp4 --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
# 使用bmcv做前处理
$ python3 retinaface_sophon_bmcv.py --bmodel ../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel --network mobile0.25 --input ../data/images/dance.mp4 --tpu_id 0 --conf 0.02 --nms 0.3 --use_np_file_as_input False
```
测试结束后会将预测图片保存至result_imgs目录下，并打印相关测试时间。

## 3. C++ demo usage

### 3.1 x86平台SC5
- 编译

```bash
$ cd cpp
$ make -f Makefile.pcie # 生成face_test
```

- 测试
```bash
# 图片模式，1batch，fp32
# imagelist.txt的每一行是图片的路径
# 如果模型是多batch的，会每攒够batch数的图片做一次推理
$ ./face_test 0 imagelist.txt ../data/models/face_detection_fp32_b1.bmodel

# 视频模式，4batch，int8
# videolist.txt的每一行是一个mp4视频路径或者一个rtsp url
# videolist.txt的视频数和模型的batch数相等
$ ./face_test 1 videolist.txt ../data/models/face_detection_int8_b4.bmodel
```
执行完毕后，会在当前目录生成一个名为result_imgs的文件夹，里面可以看到结果图片。

### 3.2 arm平台SE5

对于arm平台SE5，需要在docker开发容器中使用交叉编译工具链编译生成可执行文件，而后拷贝到Soc目标平台运行。

- 在docker开发容器中交叉编译
```bash
$ cd cpp
$ make -f Makefile.arm # 生成face_test
```

- 将生成的可执行文件及所需的模型和测试图片或视频文件拷贝到盒子中测试，测试命令同上。