import sophon.sail as sail
import numpy as np
import os
import argparse

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_opt():
    parser = argparse.ArgumentParser(description="Demo for Yolox")
    parser.add_argument('--bmodel_path', type=str, default="/workspace/test/YOLOX/models/yolox_s_fp32_batch1/compilation.bmodel")
    parser.add_argument('--feature_savepath', type=str, default="torch_random")
    parser.add_argument('--max_error',type=float,default=0.00001)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_opt()
    feature_path = args.feature_savepath
    bmodel_file = args.bmodel_path
    seed_file = os.path.join(feature_path,"seed.txt")
    if not os.path.exists(seed_file):
        print("Can not found seed_file!")
        exit(1)
    if not os.path.exists(bmodel_file):
        print("Can not open bmodel: {}".format(bmodel_file))
    with open(seed_file,"r+") as fp:
        random_seed = int(fp.readline().rstrip('\n'))
        print(random_seed)
        fp.close

    engine = sail.Engine(args.bmodel_path,0, sail.IOMode.SYSIO)
    graph_name = engine.get_graph_names()[0]

    input_name = engine.get_input_names(graph_name)[0]
    output_name = engine.get_output_names(graph_name)

    np.random.seed(random_seed)

    for idx in range(10):
        torch_out_name = os.path.join(feature_path,"{}_{}.npy".format(random_seed,idx))
        if not os.path.exists(torch_out_name):
            print("Load {} failed!".format(torch_out_name))
            exit(1)
        input_npy = np.random.rand(1,3,640,640).astype(np.float32)
        # print(input_npy)
        input_tensors = {input_name: input_npy}
        outputs = engine.process(graph_name, input_tensors) 
        output_npy = outputs[output_name[0]]
        torch_out_npy = np.load(torch_out_name)
        sub_out = abs(output_npy - torch_out_npy)

        sub_basic = sub_out.copy()
        sub_basic[:] = args.max_error
        if (sub_out <= sub_basic).all() is False:
            print("The conversion result exceeds the maxinum error!")
            exit(1)
        print("True...")
    
    print("Verification successed!")




    

