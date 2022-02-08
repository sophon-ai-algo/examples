# calibration_use_pb \
#     release \
#     -model=/workspace/YOLOX/models/yolox_s/yolox_s_bmnetp_test_fp32.prototxt \
#     -weights=/workspace/YOLOX/models/yolox_s/yolox_s_bmnetp.fp32umodel \
#     -iterations=100 \
#     -bitwidth=TO_INT8

calibration_use_pb \
    quantize \
    -model=/workspace/YOLOX/models/yolox_m/yolox_m_bmnetp_test_fp32.prototxt \
    -weights=/workspace/YOLOX/models/yolox_m/yolox_m_bmnetp.fp32umodel \
    -iterations=100 \
    -bitwidth=TO_INT8

    