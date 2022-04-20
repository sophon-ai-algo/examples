# multi_demo说明

multi这个示例工程，演示了如何进行`多模型并行`；如要参考`多模型串行`，请查看`face_recognition`例子。

**本示例与其他demo不同，使用`examples/cameras_v1.json`，作为配置文件格式**
```json
"models": [
      {
        "class_threshold": 0.5,
        "obj_threshold": 0.5,
        "nms_threshold": 0.5,
        "name": "coco_int8_4b",
        "skip_frame": 2,
        "path": "/data/workspace/models/yolov5s_4b_int8_v21.bmodel"
      },
      {
        "class_threshold": 0.5,
        "obj_threshold": 0.5,
        "nms_threshold": 0.5,
        "name": "coco_fp32_1b",
        "skip_frame": 2,
        "path": "/data/workspace/models/yolov5s.bmodel"
      }
   ]
```
> **models** 列表配置了需要并行的模型信息，包括置信度、跳帧、模型路径等。
其中`name`字段是该模型的唯一标识，切勿重复。

另外，cameras节点中models的值，就是引用上述的`name`字段
```json
"cards": [
      {
        "devid": 0,
        "cameras": [
          {
            "address": "rtsp://admin:hk123456@11.73.12.22",
            "chan_num": 2,
            "models": ["coco_int8_4b", "coco_fp32_1b"]
          }
        ]
      }
]
```
> 上述配置可以理解为
> 1. 生成coco_int8_4b、coco_fp32_1b两个模型的pipeline
> 2. 每个模型的pipeline检测chan_num=2股流
> 3. 最后进程拉取的流数量 = chan_num * models数，即4股
