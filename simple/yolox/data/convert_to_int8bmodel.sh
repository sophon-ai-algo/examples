bmnetu -model /workspace/YOLOX/models/yolox_m/yolox_m_bmnetp_deploy_int8_unique_top.prototxt \
     -weight /workspace/YOLOX/models/yolox_m/yolox_m_bmnetp.int8umodel \
     -max_n 4 \
     -prec=INT8 \
     -dyn=0 \
     -cmp=1 \
     -target=BM1684 \
     -outdir=/workspace/YOLOX/models/yolox_m/int8model
