# -*- coding: utf-8 -*- 
import os
import cv2
import numpy as np
import argparse
import sophon.sail as sail
import math
import copy
from PIL import Image, ImageDraw, ImageFont
import logging
logging.basicConfig(level=logging.DEBUG)

import det_cv_cv_sail as predict_det
import rec_cv_cv_sail as predict_rec
import cls_cv_cv_sail as predict_cls

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.PPOCRv2Det(args)
        self.text_recognizer = predict_rec.PPOCRv2Rec(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.PPOCRv2Cls(args)
        
        self.drop_score = args.drop_score
        self.crop_image_res_index = 0
        self.args = args

    def __call__(self, img_list, cls=True):
        ori_img_list = img_list.copy()
        dt_boxes_list = self.text_detector(img_list)
        #if dt_boxes is None:
        #    return None, None
        img_dict = {"imgs":[], "dt_boxes":[], "pic_ids":[]}
        for id, dt_boxes in enumerate(dt_boxes_list):
            #img_crop_list = []
            #dt_boxes = sorted_boxes(dt_boxes)
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = get_rotate_crop_image(ori_img_list[id], tmp_box)
                img_dict["imgs"].append(img_crop)
                img_dict["dt_boxes"].append(dt_boxes[bno])
                img_dict["pic_ids"].append(id)
                #img_crop_list.append(img_crop)

        if self.use_angle_cls and cls:
            img_dict["imgs"], cls_res = self.text_classifier(img_dict["imgs"])

        rec_res = self.text_recognizer(img_dict["imgs"])

        results_list = [{"dt_boxes":[], "text":[], "score":[]} for i in range(len(img_list))]
        for i, id in enumerate(rec_res.get("ids")):
            text, score = rec_res["res"][i]
            if score >= self.drop_score:
                pic_id = img_dict["pic_ids"][id]
                results_list[pic_id]["dt_boxes"].append(img_dict["dt_boxes"][id])
                results_list[pic_id]["text"].append(text)
                results_list[pic_id]["score"].append(score)

        return results_list

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img   

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="data/images/ppocr_img/fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

def main(opt):
    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    ppocrv2_sys = TextSystem(opt)
    img_list = []
    for img_name in os.listdir(opt.img_path):
        #logging.info(img_name)
        #label = img_name.split('.')[0]
        img_file = os.path.join(opt.img_path, img_name)
        #print(img_file, label)
        src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        img_list.append(src_img)

    results_list = ppocrv2_sys(img_list)    

    for i, result in enumerate(results_list):
        img_name = os.listdir(opt.img_path)[i]
        logging.info(img_name)
        logging.info(result["text"])
        image_file = os.path.join(opt.img_path, img_name)
        image = Image.fromarray(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))
        draw_img = draw_ocr_box_txt(
                image,
                result["dt_boxes"],
                result["text"],
                result["score"],
                drop_score=opt.drop_score)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "ocr_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, draw_img[:, :, ::-1])
        logging.info("The visualized image saved in {}".format(img_path))


def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--img_path', type=str, default='data/images/ppocr_img/test', help='input image path')
    parser.add_argument('--tpu_id', type=int, default=0, help='tpu id')
    # params for text detector
    parser.add_argument('--det_model', type=str, default='data/models/fp32bmodel/ch_PP-OCRv2_det_fp32_b1b4.bmodel', help='detector bmodel path')
    parser.add_argument("--det_batch_size", type=int, default=4)
    parser.add_argument('--det_limit_side_len', type=int, default=[960])
    # params for text recognizer
    parser.add_argument('--rec_model', type=str, default='data/models/fp32bmodel/ch_PP-OCRv2_rec_fp32_b1b4.bmodel', help='recognizer bmodel path')
    parser.add_argument('--img_size', type=int, default=[[320, 32],[640, 32],[1280, 32]], help='inference size (pixels)')
    parser.add_argument("--rec_batch_size", type=int, default=4)
    parser.add_argument("--char_dict_path", type=str, default="ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)
    # params for text classifier
    parser.add_argument("--use_angle_cls", type=bool, default=True)
    parser.add_argument('--cls_model', type=str, default='data/models/fp32bmodel/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel', help='classifier bmodel path')
    parser.add_argument("--cls_batch_size", type=int, default=4)
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
