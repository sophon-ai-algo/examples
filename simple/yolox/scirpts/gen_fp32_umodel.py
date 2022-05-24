import os
import ufw.tools as tools
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for resize")
    parser.add_argument('--trace_model', type=str, default="/workspace/test/YOLOX/models/yolox_s.trace.pt")
    parser.add_argument('--data_path',type=str, default="/workspace/test/YOLOX/datasets/ost_data_enhance/data.mdb")
    parser.add_argument('--dst_width',type=int, default=640)
    parser.add_argument('--dst_height',type=int, default=640)

    opt = parser.parse_args()

    save_path_temp = opt.trace_model.split('/')[-1].split('.')[0]
    save_path = os.path.join(opt.trace_model.split(save_path_temp)[0],save_path_temp)
    ptyolox = [
    '-m', '{}'.format(opt.trace_model),
    '-s', '(1,3,{},{})'.format(opt.dst_width,opt.dst_height),
    '-d', '{}'.format(save_path),
    '-D', '{}'.format(opt.data_path),
    '--cmp'
]

    tools.pt_to_umodel(ptyolox)