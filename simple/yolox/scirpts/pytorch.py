import os
import torch
import numpy as np
import argparse
import time 

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_opt():
    parser = argparse.ArgumentParser(description="Demo for Yolox")
    parser.add_argument('--model_path', type=str, default="/workspace/test/YOLOX/models/yolox_s.trace.pt")
    parser.add_argument('--feature_savepath', type=str, default="torch_random")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_opt()
    model = torch.jit.load(args.model_path)
    random_seed = int(time.time())
    mkdir(args.feature_savepath)
    seed_file = os.path.join(args.feature_savepath,"seed.txt")
    with open(seed_file,"w+") as fp:
        fp.write(str(random_seed))
        fp.close()
    np.random.seed(random_seed)
    for idx in range(10):
        input_npy = np.random.rand(1,3,640,640).astype(np.float32)
        input_tensor = torch.from_numpy(input_npy)
        output_tensor = model(input_tensor)
        output_npy = output_tensor.detach().numpy()
        save_name = os.path.join(args.feature_savepath,"{}_{}".format(random_seed,idx))
        np.save(save_name,output_npy)



    


