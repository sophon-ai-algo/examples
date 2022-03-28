# -*- coding: utf-8 -*- 

import os
import cv2
import numpy as np
import argparse
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.DEBUG)


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {i:char for i, char in enumerate(CHARS)}

# input: x.1, [1, 3, 24, 96], float32, scale: 1
class LPR(object):
    def __init__(self, opt):
        # load bmodel
        model_path = opt.bmodel
        print("using model {}".format(model_path))
        self.net = sail.Engine(model_path, opt.tpu_id, sail.IOMode.SYSIO)
        print("load bmodel success!")
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]

        self.input_shape = [1, 3, 24, 94]
        self.input_shapes = {self.input_name: self.input_shape}
        self.output_shape = [1, 1, 68, 18]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)

        # get handle to create input and output tensors  
        self.handle = self.net.get_handle()
        self.input = sail.Tensor(self.handle, self.input_shape,  self.input_dtype,  False, False)
        self.output = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True,  True)
        self.input_tensors = {self.input_name: self.input}
        self.output_tensors = {self.output_name: self.output}
        # set io_mode
        # self.net.set_io_mode(self.graph_name, sail.IOMode.SYSO)
        # init bmcv for preprocess
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        self.scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.ab = [x * self.scale * 0.0078125 for x in [1, -127.5, 1, -127.5, 1, -127.5]]

        
    def decode_bmcv(self, img_file):
        decoder = sail.Decoder(img_file)
        img = decoder.read(self.handle)
        return img

    def preprocess_bmcv(self, img):
        output = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3], \
                        sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)

        tmp = self.bmcv.vpp_resize(img, 94, 24)
        
        self.bmcv.convert_to(tmp, output, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))
        self.bmcv.bm_image_to_tensor(output, self.input)

    def predict(self):
        self.net.process(self.graph_name, self.input_tensors, self.input_shapes, self.output_tensors)
        real_output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        outputs = self.output.asnumpy(real_output_shape)
        return outputs

    def postprocess(self, outputs):
        res = list()
        #outputs = list(outputs.values())[0]
        outputs = np.argmax(outputs, axis = 1)
        for output in outputs:
            no_repeat_blank_label = list()
            pre_c = output[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(CHARS_DICT[pre_c])
            for c in output:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(CHARS_DICT[c])
                pre_c = c
            res.append(''.join(no_repeat_blank_label)) 

        return res

    def process(self, img_file):
        img = self.decode_bmcv(img_file)
        self.preprocess_bmcv(img)
        outputs = self.predict()
        res = self.postprocess(outputs)
        return res

def main(opt):
    lpr = LPR(opt)
    if os.path.isfile(opt.img_path):
        res = lpr.process(opt.img_path)
        logging.info("img:{}, res:{}".format(opt.img_path, res[0]))
    else:
        Tp = 0
        for img_name in os.listdir(opt.img_path):
            if opt.mode == 'val':
                label = img_name.split('.')[0]
            img_file = os.path.join(opt.img_path, img_name)
            res = lpr.process(img_file)
            logging.info("img:{}, res:{}".format(img_file, res[0]))
            if opt.mode == 'val' and res[0] == label:
                Tp += 1
        if opt.mode == 'val':
            cn = len(os.listdir(opt.img_path))
            logging.info("ACC = %.4f" % (Tp / cn))

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    #parser.add_argument('--img-size', type=int, default=[94, 24], help='inference size (pixels)')
    parser.add_argument('--mode', type=str, default='val')
    parser.add_argument('--img_path', type=str, default='/workspace/projects/LPRNet/data/images/test', help='input image path')
    parser.add_argument('--bmodel', type=str, default='/workspace/projects/LPRNet/scripts/fp32model/compilation.bmodel', help='input model path')
    parser.add_argument('--tpu_id', type=int, default=0, help='tpu id')
    #parser.add_argument('--format', type=str, default="fp32", help='model format fp32 or fix8b')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
