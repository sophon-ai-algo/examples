# -*- coding: utf-8 -*- 

import os
import time
import cv2
import numpy as np
import argparse
#from pyrsistent import T
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
    def __init__(self, opt, img_size = [94, 24]):
        # load bmodel
        model_path = opt.bmodel
        print("using model {}".format(model_path))
        self.net = sail.Engine(model_path, opt.tpu_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        #self.scale = self.net.get_input_scale(self.graph_name, self.input_name)
        #self.scale *= 0.0078125
        print("load bmodel success!")
        self.img_size = img_size
        self.dt = 0.0

    def preprocess(self, img):
        h, w, _ = img.shape
        if h != self.img_size[1] or w != self.img_size[0]:
            img = cv2.resize(img, self.img_size)
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))


        # CHW to NCHW format
        img = np.expand_dims(img, axis=0)
        # Convert the img to row-major order, also known as "C order":
        #img = np.ascontiguousarray(img)

        return img

    def predict(self, tensor):
        input_data = {self.input_name: np.array(tensor, dtype=np.float32)}
        t0 = time.time()
        outputs = self.net.process(self.graph_name, input_data)
        self.dt += time.time() - t0
        return list(outputs.values())[0]

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

    def process(self, img):
        img = self.preprocess(img)
        outputs = self.predict(img)
        res = self.postprocess(outputs)
        return res

    def get_time(self):
        return self.dt

def main(opt):
    lpr = LPR(opt)
    if os.path.isfile(opt.img_path):
        src_img = cv2.imdecode(np.fromfile(opt.img_path, dtype=np.uint8), -1)
        res = lpr.process(src_img)
        logging.info("img:{}, res:{}".format(opt.img_path, res[0]))
    else:
        Tp = 0
        for img_name in os.listdir(opt.img_path):
            if opt.mode == 'val':
                label = img_name.split('.')[0]
            img_file = os.path.join(opt.img_path, img_name)
            src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
            res = lpr.process(src_img)
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
