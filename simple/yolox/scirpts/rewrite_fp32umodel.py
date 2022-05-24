import os
import argparse

from cv2 import line

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for rewrite")
    parser.add_argument('--trace_model', type=str, default="/workspace/test/YOLOX/models/yolox_s.trace.pt")

    opt = parser.parse_args()

    save_path_temp = opt.trace_model.split('/')[-1].split('.')[0]
    fp32_umodel_path = os.path.join(opt.trace_model.split(save_path_temp)[0],save_path_temp)
    opt = parser.parse_args()
    file_list = os.listdir(fp32_umodel_path)
    for file_name in file_list:
        if file_name[-13:] != 'fp32.prototxt':
            continue
        file_name_sa = os.path.join(fp32_umodel_path,file_name)
        with open(file_name_sa,"r+") as fp:
            lines = fp.readlines()
            last_line = lines[-1]
            lines[-1] = "  forward_with_float:true\n"
            fp.close()
            with open(file_name_sa,"w+") as fp_save:
                fp_save.writelines(lines)
                fp_save.write(last_line)
                fp_save.close()
