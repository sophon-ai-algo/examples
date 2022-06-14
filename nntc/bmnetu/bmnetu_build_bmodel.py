import os, shutil
import bmnetu

model = r'models/mobilenet/mobilenet_deploy_int8_unique_top.prototxt' # 模型文件
weight = r'models/mobilenet/mobilenet.int8umodel' #权重文件
target = r"BM1684" #目标设备
export_dir = r"python-output/mobilenet" #输出目录

if os.path.exists(export_dir):
  shutil.rmtree(export_dir)
os.makedirs(export_dir)
bmnetu.compile(model = model,weight = weight, target = target, outdir = export_dir, net_name='mobilenet-int8')
