#Retinaface Face Detect Demos

data目录，包含了测试图片及模型
python目录, 使用opencv或bmcv做前处理，sail作推理，numpy做后处理
cpp目录，cpp的示例程序

注意：cpp和python示例程序所用的模型结构不同
cpp代码对应模型：face_detection_fp32_b1/b4.bmodel，对应9个输出tensor
python代码对应模型：retinaface_***_384x640_fp32_b1/b4.bmodel，对应3个输出tensor

> 已经转换好的bmodel文件可从以下百度网盘下载：
> 链接: https://pan.baidu.com/s/1d3f8CjzC3BF2-2I2OF0q1g 提取码: lt59 
