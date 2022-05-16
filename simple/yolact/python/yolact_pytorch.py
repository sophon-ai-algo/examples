import os
import shutil
import torch
import numpy as np
import cv2
import argparse
import configparser
from utils.preprocess_numpy import PreProcess
from utils.postprocess_numpy import PostProcess
from utils.utils import draw_numpy, is_img

class Detector(object):
    def __init__(self, cfg_path, tstrace_path,
                 conf_thresh=0.5, nms_thresh=0.5, keep_top_k=200):
        try:
            self.get_config(cfg_path)
        except Exception as e:
            raise e

        if not os.path.exists(tstrace_path):
            raise FileNotFoundError('{} is not existed.'.format(tstrace_path))
        self.net = torch.jit.load(tstrace_path)
        self.net.eval()
        print('{} is loaded.'.format(tstrace_path))

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k

        self.preprocess = PreProcess(self.cfg)
        self.postprocess = PostProcess(
            self.cfg,
            self.conf_thresh,
            self.nms_thresh,
            self.keep_top_k,
        )


    def get_config(self, cfg_path):
        if not os.path.exists(cfg_path):
            raise FileNotFoundError('{} is not existed.'.format(cfg_path))

        config = configparser.ConfigParser()
        config.read(cfg_path)

        normalize = config.get("yolact", "normalize")
        subtract_means = config.get("yolact", "subtract_means")
        to_float = config.get("yolact", "to_float")

        width = config.get("yolact", "width")
        height = config.get("yolact", "height")
        conv_ws = config.get("yolact", "conv_ws")
        conv_hs = config.get("yolact", "conv_hs")
        aspect_ratios = config.get("yolact", "aspect_ratios")
        scales = config.get("yolact", "scales")
        variances = config.get("yolact", "variances")

        self.cfg = dict()

        self.cfg['normalize'] = int(normalize.split(',')[0])
        self.cfg['subtract_means'] = int(subtract_means.split(',')[0])
        self.cfg['to_float'] = int(to_float.split(',')[0])
        self.cfg['width'] = int(width.split(',')[0])
        self.cfg['height'] = int(height.split(',')[0])
        self.cfg['conv_ws'] = [int(i) for i in conv_ws.replace(' ', '').split(',')]
        self.cfg['conv_hs'] = [int(i) for i in conv_hs.replace(' ', '').split(',')]
        self.cfg['aspect_ratios'] = [float(i) for i in aspect_ratios.replace(' ', '').split(',')]
        self.cfg['scales'] = [int(i) for i in scales.replace(' ', '').split(',')]
        self.cfg['variances'] = [float(i) for i in variances.replace(' ', '').split(',')]


    @torch.no_grad()
    def predict(self, tensor):
        if tensor.ndim != 4:
            tensor = np.expand_dims(tensor, 0)
        input_tensor = torch.from_numpy(tensor)
        out = self.net(input_tensor)
        return [e_out.cpu().numpy() for e_out in out]


def decode_image_opencv(image_path):
    try:
        with open(image_path, "rb") as f:
            image = np.array(bytearray(f.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        image = None
    return image


def main(opt):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    else:
        shutil.rmtree(opt.output_dir)
        os.makedirs(opt.output_dir)

    yolact = Detector(
        opt.cfgfile,
        opt.model,
        conf_thresh=opt.thresh,
        nms_thresh=opt.nms,
        keep_top_k=opt.keep,
    )

    batch_size = opt.batch_size
    input_path = opt.input_path

    # imgage directory
    input_list = []
    if os.path.isdir(input_path):
        for img_name in os.listdir(input_path):
            if is_img(img_name):
                input_list.append(os.path.join(input_path, img_name))
                # imgage file
    elif is_img(input_path):
        input_list.append(input_path)
    # imgage list saved in file
    else:
        with open(input_path, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                line_head = line.strip("\n").split(' ')[0]
                if is_img(line_head):
                    input_list.append(line_head)

    img_num = len(input_list)
    batch_num = batch_size

    # combine into batch
    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        inp_batch = []
        cur_bs = end_img_no - beg_img_no
        padding_bs = batch_num - cur_bs

        for ino in range(beg_img_no, end_img_no):
            inp_batch.append(input_list[ino])
            # padding batch for last batch
            if ino == end_img_no - 1:
                for pbs in range(padding_bs):
                    inp_batch.append(input_list[0])

        # decode
        images = []
        batch_ret = True
        for i in range(len(inp_batch)):
            image = decode_image_opencv(inp_batch[i])
            if image is None:
                batch_ret = False
                break
            images.append(image)
        # if one decode failed, pass this batch
        if not batch_ret:
            continue

        org_size_list = []
        for i in range(len(inp_batch)):
            org_h, org_w = images[i].shape[:2]
            org_size_list.append((org_w, org_h))

        # batch end-to-end inference
        preprocessed_img = yolact.preprocess.infer_batch(images)

        out_infer = yolact.predict(preprocessed_img)

        classid_list, conf_scores_list, boxes_list, masks_list = \
            yolact.postprocess.infer_batch(out_infer, org_size_list)

        for i, (e_img, classid, conf_scores, boxes, masks) in enumerate(zip(images,
                                                                            classid_list,
                                                                            conf_scores_list,
                                                                            boxes_list,
                                                                            masks_list)):
            draw_numpy(e_img, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)
            save_basename = 'res_pt_{}'.format(os.path.basename(inp_batch[i]))
            save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
            cv2.imencode('.jpg', e_img)[1].tofile('{}.jpg'.format(save_name))
    print('the results is saved: {}'.format(os.path.abspath(opt.output_dir)))


def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--cfgfile', type=str, help='model config file')
    parser.add_argument('--model', type=str, help='torchscript trace model path')
    image_path = os.path.join(os.path.dirname(__file__),"../data/images/zidane.jpg")
    parser.add_argument('--thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--keep', type=int, default=100, help='keep top-k')
    parser.add_argument('--input_path', type=str, default=image_path, help='input image path')
    parser.add_argument('--output_dir', type=str, default="results_pt", help='output image directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
