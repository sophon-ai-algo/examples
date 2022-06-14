# 工具链使用示例

## 0. 初始化SOPHONSDK环境，请参考SOPHONSDK使用文档

## 1.各种编译前端使用示例
编译前端提供了命令行和python调用两种方式编译模型，可以根据业务需要或习惯自行选择其中一种即可
这里仅给出了使用的基本编译示例，具体详情可参考[官网](https://developer.sophgo.com/site/index/document/2/all.html)的NNTOOLCHAIN文档。

### 1.1 使用bmnetc编译caffe模型
``` shell
cd bmnetc                                   # 进入示例的bmnetc目录
./bmnetc_build_bmodel.sh                    # 命令行方式调用bmnetc编译模型
bmrt_test --context_dir=output/det2/        # 测试编译出的bmodel是否正确

python3 bmnetc_build_bmodel.py              # 利用python脚本调用bmnetc编译模型
bmrt_test --context_dir=python-output/det2/ # 测试编译出的bmodel是否正确

cd ..
```

### 1.2 使用bmnett编译tensorflow的frozen后的模型(.pb文件)

``` shell
cd bmnett                                     # 进入示例的bmnett目录

./bmnett_build_bmodel.sh                      # 命令行方式调用bmnett编译模型
bmrt_test --context_dir=output/vqvae/         # 测试编译出的bmodel是否正确

python3 bmnett_build_bmodel.py                # 利用python脚本调用bmnett编译模型
bmrt_test --context_dir=python-output/vqvae/  # 测试编译出的bmodel是否正确

cd ..
```

### 1.3 使用bmnetp编译pytorch的trace后的模型(通常以pt为后缀)
``` shell
cd bmnetp                                       # 进入示例的bmnetp目录

./bmnetp_build_bmodel.sh                        # 命令行方式调用bmnetp编译模型
bmrt_test --context_dir=output/anchors/         # 测试编译出的bmodel是否正确

python3 bmnetp_build_bmodel.py                  # 利用python脚本调用bmnetp编译模型
bmrt_test --context_dir=python-output/anchors/  # 测试编译出的bmodel是否正确

cd ..
```

### 1.4 使用bmneto编译onnx模型
``` shell
cd bmneto                                       # 进入示例的bmneto目录

./bmneto_build_bmodel.sh                        # 命令行方式调用bmneto编译模型
bmrt_test --context_dir=output/yolov5s/         # 测试编译出的bmodel是否正确

python3 bmneto_build_bmodel.py                  # 利用python脚本调用bmneto编译模型
bmrt_test --context_dir=python-output/yolov5s/  # 测试编译出的bmodel是否正确

cd ..
```
### 1.5 使用bmpaddle编译PaddlePaddle模型
``` shell
cd bmpaddle                                               # 进入示例的bmpaddle目录

./bmpaddle_build_bmodel.sh                                # 命令行方式调用bmpaddle编译模型
bmrt_test --context_dir=output/ch_ppocr_mobile_v2/        # 测试编译出的bmodel是否正确

python3 bmpaddle_build_bmodel.py                          # 利用python脚本调用bmpaddle编译模型
bmrt_test --context_dir=python-output/ch_ppocr_mobile_v2/ # 测试编译出的bmodel是否正确

cd ..
```

### 1.6 使用bmnetm编译MxNet模型
``` shell
cd bmnetm                                     # 进入示例的bmnetm目录

./bmnetm_build_bmodel.sh                      # 命令行方式调用bmnetm编译模型
bmrt_test --context_dir=output/lenet/         # 测试编译出的bmodel是否正确

python3 bmnetm_build_bmodel.py                # 利用python脚本调用bmnetm编译模型
bmrt_test --context_dir=python-output/lenet/  # 测试编译出的bmodel是否正确

cd ..
```

### 1.7 使用bmnetd编译Darknet模型
``` shell
cd bmnetd                                           # 进入示例的bmnetd目录

./bmnetd_build_bmodel.sh                            # 命令行方式调用bmnetd编译模型
bmrt_test --context_dir=output/yolov3-tiny/         # 测试编译出的bmodel是否正确

python3 bmnetd_build_bmodel.py                      # 利用python脚本调用bmnetd编译模型
bmrt_test --context_dir=python-output/yolov3-tiny/  # 测试编译出的bmodel是否正确

cd ..
```

### 1.8 使用bmnetu编译int8的量化umodel模型
umodel是我们量化工具最终生成的int8的量化模型

``` shell
cd bmnetu                                         # 进入示例的bmnetd目录

./bmnetu_build_bmodel.sh                          # 命令行方式调用bmnetu编译模型
bmrt_test --context_dir=output/mobilenet/         # 测试编译出的bmodel是否正确

python3 bmnetu_build_bmodel.py                    # 利用python脚本调用bmnetu编译模型
bmrt_test --context_dir=python-output/mobilenet/  # 测试编译出的bmodel是否正确

cd ..
```
## 2.运行时编程及使用示例

该示例是一个分类程序，演示了runtime的基本用法和流程
该应用需要在pcie模式下运行

```shell
cd bmnetc && ./bmnetc_build_bmodel.sh && cd ..            #编译bmnetc模型, 生成分类所需的bmodel
cd runtime/classify/
make                                                      #编译应用程序，生成classify应用
./classify ../../bmnetc/output/det2 ../test_fig/guys0.jpg #执行程序
cd ..
```
