import os, shutil
import bmnetu

model = r'models/mobilenet/mobilenet_deploy_int8_unique_top.prototxt'
weight = r'models/mobilenet/mobilenet.int8umodel'
target = r"BM1684"
export_dir = r"python-output/mobilenet"

if os.path.exists(export_dir):
  shutil.rmtree(export_dir)
os.makedirs(export_dir)
bmnetu.compile(model = model, weight = weight, target = target, outdir = export_dir, net_name='mobilenet-int8')
