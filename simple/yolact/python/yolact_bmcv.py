import os
import shutil
import torch
import numpy as np
import cv2
import argparse
import configparser
from utils.preprocess_bmcv import PreProcess
from utils.postprocess_numpy import PostProcess
from utils.sophon_inference import SophonInference
import sophon.sail as sail
from utils.utils import draw_bmcv, draw_numpy, is_img


class Detector(object):
    def __init__(self, cfg_path, bmodel_path, device_id=0,
                 conf_thresh=0.5, nms_thresh=0.5, keep_top_k=200):
        try:
            self.get_config(cfg_path)
        except Exception as e:
            raise e

        if not os.path.exists(bmodel_path):
            raise FileNotFoundError('{} is not existed.'.format(bmodel_path))
        self.net = SophonInference(model_path=bmodel_path,
                                   device_id=device_id)
        print('{} is loaded.'.format(bmodel_path))

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k

        self.output_order_node = ['loc', 'conf', 'mask', 'proto']
        self.bmcv = self.net.bmcv
        self.handle = self.net.handle
        self.input_scale = list(self.net.input_scales.values())[0]

        self.preprocess = PreProcess(self.cfg, self.input_scale)
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

    def predict(self, tensor):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            tensor:

        Returns:

        """
        # feed: [input0]

        out_dict = self.net.infer_bmimage(tensor)
        # resort
        out_keys = list(out_dict.keys())
        ord = []
        for n in self.output_order_node:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [out_dict[out_keys[i]] for i in ord]
        return out


def decode_image_bmcv(image_path, process_handle, img):
    # img = sail.BMImage()
    # img = sail.BMImageArray4D()
    decoder = sail.Decoder(image_path, True, 0)
    if isinstance(img, sail.BMImage):
        ret = decoder.read(process_handle, img)
    else:
        ret = decoder.read_(process_handle, img)
    if ret != 0:
        return False
    return True


def main(opt):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    else:
        shutil.rmtree(opt.output_dir)
        os.makedirs(opt.output_dir)

    yolact = Detector(
        opt.cfgfile,
        opt.model,
        device_id=opt.dev_id,
        conf_thresh=opt.thresh,
        nms_thresh=opt.nms,
        keep_top_k=opt.keep,
    )

    batch_size = yolact.net.inputs_shapes[0][0]
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

    if batch_size not in [1, 4]:
        raise NotImplementedError(
            'This example is for batch-1 or batch-4 model. Please transfer bmodel with batch size 1 or 4.')

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

        if len(inp_batch) == 1:

            image = sail.BMImage()
            ret = decode_image_bmcv(inp_batch[0], yolact.handle, image)
            if not ret:
                # decode failed.
                print('decode failed: {}'.format(inp_batch[0]))
                continue
            org_h, org_w = image.height(), image.width()

            # end-to-end inference
            preprocessed_img = yolact.preprocess(image,
                                                 yolact.handle,
                                                 yolact.bmcv,
                                                 )

            out_infer = yolact.predict([preprocessed_img])

            loc_data, conf_preds, mask_data, proto_data = out_infer
            classid, conf_scores, boxes, masks = \
                yolact.postprocess(loc_data, conf_preds, mask_data, proto_data, (org_w, org_h))

            # bmcv cannot draw with instance masks, so we convert BMImage to numpy to draw
            image_bgr_planar = sail.BMImage(yolact.handle, image.height(), image.width(),
                                            sail.Format.FORMAT_BGR_PLANAR, image.dtype())
            yolact.bmcv.convert_format(image, image_bgr_planar)
            image_tensor = yolact.bmcv.bm_image_to_tensor(image_bgr_planar)
            image_chw_numpy = image_tensor.asnumpy()[0]
            image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()

            draw_numpy(image_numpy, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)
            # draw_bmcv(yolact.bmcv, image, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)

            save_basename = 'res_bmcv_{}'.format(os.path.basename(inp_batch[0]))
            save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
            cv2.imencode('.jpg', image_numpy)[1].tofile('{}.jpg'.format(save_name))
            # yolact.bmcv.imwrite('{}.jpg'.format(save_name), image)
            print('{}.jpg is saved.'.format(save_name))

        elif len(inp_batch) == 4:

            image_list = []
            preprocessed_imgs = sail.BMImageArray4D(yolact.handle,
                                              yolact.preprocess.height,
                                              yolact.preprocess.width,
                                              sail.FORMAT_RGB_PLANAR,
                                              sail.DATA_TYPE_EXT_FLOAT32)
            batch_ret = True
            for i in range(len(inp_batch)):
                image = sail.BMImage()
                ret = decode_image_bmcv(inp_batch[i], yolact.handle, image)
                if not ret:
                    batch_ret = False
                    break
                image_list.append(image)
            # if one decode failed, pass this batch
            if not batch_ret:
                continue

            org_size_list = []
            for i in range(len(inp_batch)):
                org_h, org_w = image_list[i].height(), image_list[i].width()
                org_size_list.append((org_w, org_h))

            # batch end-to-end inference
            preprocessed_img_list = yolact.preprocess.infer_batch(
                image_list,
                yolact.handle,
                yolact.bmcv,
            )

            for i in range(len(inp_batch)):
                preprocessed_imgs.copy_from(i, preprocessed_img_list[i])

            out_infer = yolact.predict([preprocessed_imgs])

            classid_list, conf_scores_list, boxes_list, masks_list = \
                yolact.postprocess.infer_batch(out_infer, org_size_list)

            # cancel padding batch for last batch
            classid_list, conf_scores_list, boxes_list, masks_list = \
                classid_list[:cur_bs], conf_scores_list[:cur_bs], boxes_list[:cur_bs], masks_list[:cur_bs]

            for i, (e_img, classid, conf_scores, boxes, masks) in enumerate(zip(image_list,
                                                                                classid_list,
                                                                                conf_scores_list,
                                                                                boxes_list,
                                                                                masks_list)):
                # bmcv cannot draw with instance masks, so we convert BMImage to numpy to draw
                image_bgr_planar = sail.BMImage(yolact.handle, e_img.height(), e_img.width(),
                                                sail.Format.FORMAT_BGR_PLANAR, e_img.dtype())
                yolact.bmcv.convert_format(e_img, image_bgr_planar)
                image_tensor = yolact.bmcv.bm_image_to_tensor(image_bgr_planar)
                image_chw_numpy = image_tensor.asnumpy()[0]
                image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()

                draw_numpy(image_numpy, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)
                # draw_bmcv(yolact.bmcv, e_img, boxes, masks=masks, classes_ids=classid, conf_scores=conf_scores)

                save_basename = 'res_bmcv_{}'.format(os.path.basename(inp_batch[i]))
                save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
                cv2.imencode('.jpg', image_numpy)[1].tofile('{}.jpg'.format(save_name))
                # yolact.bmcv.imwrite('{}.jpg'.format(save_name), e_img)
            print('the results is saved: {}'.format(os.path.abspath(opt.output_dir)))


        else:
            raise NotImplementedError


def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--cfgfile', type=str, help='model config file')
    parser.add_argument('--model', type=str, help='torchscript trace model path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    image_path = os.path.join(os.path.dirname(__file__),"../data/images")
    parser.add_argument('--thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--keep', type=int, default=100, help='keep top-k')
    parser.add_argument('--input_path', type=str, default=image_path, help='input image path')
    parser.add_argument('--output_dir', type=str, default="results_bmcv", help='output image directory')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)