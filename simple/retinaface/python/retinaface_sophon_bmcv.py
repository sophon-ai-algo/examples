"""
A Retinaface demo using Sophon SAIL api to make inferences.
"""

# -*- coding: utf-8 -*
import argparse
import os
import sys
import time

import cv2
import numpy as np

import sophon.sail as sail

from data.config import cfg_mnet, cfg_re50
from utils.box_utils import PriorBox, decode, decode_landm, draw_one_on_bmimage, py_cpu_nms, draw_one
from utils.print_utils import print_info
from utils.time_utils import timeit

from loguru import logger

opt = None
save_path = os.path.join(os.path.dirname(
    __file__), "result_imgs", os.path.basename(__file__).split('.')[0])

# # 设置numpy运算精度
# np.set_printoptions(threshold=np.inf)

class Retinaface_sophon(object):
    """
    description: A Retineface class that warps Sophon ops, preprocess and postprocess ops.
    """

    def __init__(self, cfg, bmodel_file_path, tpu_id, score_threshold = 0.5, nms_threshold = 0.3):
        """
        :param cfg: retinaface使用的backbone及网络配置参数
        :param bmodel_file_path: 模型路径
        :param tpu_id: tpu序列号
        :param layers: 18 , 50
        :param score_threshold: 置信度阈值
        :param nms_threshold: nms阈值
        """
        # Create a Context on sophon device
        tpu_count = sail.get_available_tpu_num()
        logger.debug('{} TPUs Detected, using TPU {} \n'.format(tpu_count, tpu_id))
        self.engine = sail.Engine(bmodel_file_path, tpu_id, sail.IOMode.SYSIO)
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        graph_count = len(self.engine.get_graph_names())
        logger.warning("{} graphs in {}, using {}".format(graph_count, bmodel_file_path, self.graph_name))

        # create input tensors
        input_names     = self.engine.get_input_names(self.graph_name)
        input_tensors   = {}
        input_shapes    = {}
        input_scales    = {}
        input_dtypes    = {}
        inputs          = []
        input_w         = 0
        input_h         = 0

        for input_name in input_names:
            input_shape = self.engine.get_input_shape(self.graph_name, input_name)
            input_dtype = self.engine.get_input_dtype(self.graph_name, input_name)
            input_scale = self.engine.get_input_scale(self.graph_name, input_name)

            input_w = int(input_shape[-1])
            input_h = int(input_shape[-2])

            # logger.debug("[{}] create sail.Tensor for input: {} ".format(input_name, input_shape))
            input = sail.Tensor(self.handle, input_shape, input_dtype, False, False)

            inputs.append(input)
            input_tensors[input_name] = input
            input_shapes[input_name] = input_shape
            input_scales[input_name] = input_scale
            input_dtypes[input_name] = input_dtype

        # create output tensors
        output_names    = self.engine.get_output_names(self.graph_name)
        output_tensors  = {}
        output_shapes   = {}
        output_scales   = {}
        output_dtypes   = {}
        outputs         = []

        for output_name in output_names:
            output_shape = self.engine.get_output_shape(self.graph_name, output_name)
            output_dtype = self.engine.get_output_dtype(self.graph_name, output_name)
            output_scale = self.engine.get_output_scale(self.graph_name, output_name)

            # create sail.Tensor for output
            # logger.debug("[{}] create sail.Tensor for output: {} ".format(output_name, output_shape))
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)

            outputs.append(output)
            output_tensors[output_name] = output
            output_shapes[output_name] = output_shape
            output_scales[output_name] = output_scale
            output_dtypes[output_name] = output_dtype

        # Store
        self.inputs = inputs
        self.input_name = input_names[0]
        self.input_tensors = input_tensors
        self.input_scale = input_scales[input_names[0]]
        self.input_dtype = input_dtypes[input_names[0]]
        
        self.outputs = outputs
        self.output_names = output_names
        self.output_tensors = output_tensors
        self.output_shapes = output_shapes

         # since Retinaface Net has only one input, set input width and height for preprocessing to use
        self.input_w = input_w
        self.input_h = input_h

        # create bmcv handle
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        logger.info("===========================================")
        logger.info("BModel: {}".format(bmodel_file_path))
        logger.info("Input : {}, {}".format(input_shapes, input_dtypes))
        logger.info("Output: {}, {}".format(output_shapes, output_dtypes))
        logger.info("===========================================")

        self.keep_top_k = 50
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.ab_bgr = [x * self.input_scale for x in [1, -104, 1, -117, 1, -123]]
        # self.mean_bgr = (104, 117, 123)

        self.cfg = cfg
        priorbox = PriorBox(cfg, image_size=(self.input_h, self.input_w))
        self.priors = priorbox.forward()

    @timeit
    def preprocess_with_bmcv(self, bm_image):
        """
        description: preprocess the input bm_image, resize and pad it to target size,
                     normalize to [0,1],transform to NCHW format.
        param:
            bm_image: BMImage
        return:
            image:  the processed bm_image
            h: original height
            w: original width
        """

        # img_w = bm_image.width()
        # img_h = bm_image.height()
        print_info(bm_image)

        resized_img = self.bmcv.vpp_resize(bm_image, self.input_w, self.input_h)
        self.bmcv.imwrite(os.path.join(save_path, "resized_img.bmp"), resized_img)
        print_info(resized_img)

        input_img = sail.BMImage(self.handle, self.input_h, self.input_w,
                                           sail.Format.FORMAT_BGR_PLANAR, self.img_dtype)
        self.bmcv.convert_to(
            resized_img, input_img, \
            ((self.ab_bgr[0], self.ab_bgr[1]), (self.ab_bgr[2], self.ab_bgr[3]), (self.ab_bgr[4], self.ab_bgr[5])))
            
        print_info(input_img)

        return input_img

    @timeit
    def infer_bmimage(self, data, USE_NP_FILE_AS_INPUT=False):

        output_nps = []  # inference output

        if USE_NP_FILE_AS_INPUT:
            
            ref_data = np.load("./np_input.npy")
            logger.info("using numpy data as input: {}".foramt(ref_data.shape))

            input = sail.Tensor(self.handle, ref_data)
            input_tensors = {self.input_name: input}
            input_shapes = {self.input_name: self.input_shape}

            logger.debug("engine process start")
            self.engine.process(self.graph_name, input_tensors, input_shapes, self.output_tensors)
            logger.debug("engine process end")
        else:
            logger.info("using decoder data as input")
            self.bmcv.bm_image_to_tensor(data, self.inputs[0])
            logger.debug("engine process start")
            self.engine.process(self.graph_name, self.input_tensors, self.output_tensors)
            logger.debug("engine process end")
        
        # convert output tensor to numpy, 遍历output_names过程中会将输出按照名称排序      
        output_nps = [output_tensor.asnumpy(self.output_shapes[output_name]) \
            for output_name, output_tensor in self.output_tensors.items()]

        return output_nps

    @timeit
    def postprocess(self, outputs, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,x1,y1,x2,y2,conf,landmark_x1,landmark_y1,
            landmark_x2,landmark_y2,...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """

        logger.debug("outputs size: {}".format(len(outputs)))
        # [1, 25500, 4], bbox, xywh
        # [1, 25500, 10] landmarks, x1,y1,x2,y2,x3,y3,x4,y4,x5,y5
        # [1, 25500, 2]  conf

        logger.debug("output tensor 0 = {} , output tensor 1 = {}, output tensor 2 = {} ".format(
            outputs[0].shape, outputs[1].shape, outputs[2].shape))

        # 根据shape取出相应的tensor
        for i in range(3):
            if outputs[i].shape[-1] == 2:
                conf = outputs[i]
            elif outputs[i].shape[-1] == 4:
                loc = outputs[i]
            else:
                landms = outputs[i] 

        # j = 0
        # loc = loc[j,...]   # x y w h
        # loc = np.expand_dims(loc, 0)
        # landms = landms[j,...]  
        # landms = np.expand_dims(landms, 0)
        # conf = conf[j,...]
        # conf = np.expand_dims(conf, 0)

        logger.debug("loc = {} , landms = {}, conf = {} ".format(loc.shape, landms.shape, conf.shape))
        
        # 解码
        scale = np.array([origin_w, origin_h, origin_w, origin_h])
        boxes = decode(loc.squeeze(0), self.priors, self.cfg['variance'])
        boxes = boxes * scale

        scores = conf.squeeze(0)[:, 1]

        landms = decode_landm(landms.squeeze(0), self.priors, self.cfg['variance'])
        scale1 = np.array([origin_w, origin_h, origin_w, origin_h,
                           origin_w, origin_h, origin_w, origin_h,
                           origin_w, origin_h])
        landms = landms * scale1

        logger.debug("after output decode: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

        # 根据置信度过滤, 减少后续运算量
        inds = np.where(scores >= self.score_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]
        
        logger.debug("after threshold filter: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.keep_top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        logger.debug("after keep-topk: boxes = {} , landmarks = {} , scores = {} ".format(boxes.shape, landms.shape, scores.shape))

        # do NMS
        boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(boxes, self.nms_threshold)
        result_boxes = boxes[keep, :]
        result_landmarks = landms[keep, :]
        logger.debug("after nms: result_boxes = {} , result_landmarks = {} ".format( \
            result_boxes.shape, result_landmarks.shape))
        # dets = np.concatenate((result_boxes, result_landmarks), axis=1)

        return result_boxes, result_landmarks

    @timeit
    def predict_bmimage(self, frame):

        if not isinstance(frame, type(None)):

            # Do image preprocess
            img = self.preprocess_with_bmcv(frame)

            # Do inference
            outputs = self.infer_bmimage(img)

            # Do postprocess
            result_boxes, result_landmarks = self.postprocess(
                outputs, frame.height(), frame.width()
            )

            logger.info("Detected {} faces ".format(len(result_boxes)))

            # Draw rectangles and labels on the original image
            result_image = frame

            # Save image
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                landmark = result_landmarks[i]

                logger.debug("face {}: x1, y1, x2, y2, conf = {}".format(i, box))
                
                draw_one_on_bmimage(
                    self.bmcv,
                    box,
                    landmark,
                    result_image,
                    label="{}:{:.2f}".format( 'Face', box[4]))
            
            return result_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Demo of Retinaface with preprocess by BMCV")

    parser.add_argument('--bmodel',
                        type=str,
                        default="../data/models/retinaface_mobilenet0.25_384x640_fp32_b1.bmodel",
                        required=False,
                        help='bmodel file path.')

    parser.add_argument('--network',
                        type=str,
                        default="mobile0.25",
                        required=False,
                        help='backbone network type: mobile0.25 , resnet50.')
    
    parser.add_argument('--input',
                        type=str,
                        default="../data/images/face1.jpg",
                        required=False,
                        help='input pic/video file path.')

    parser.add_argument('--tpu_id',
                        default=0,
                        type=int,
                        required=False,
                        help='tpu dev id(0,1,2,...).')

    parser.add_argument("--conf",
                        default=0.02,
                        type=float,
                        help="test conf threshold.")

    parser.add_argument("--nms",
                        default=0.3,
                        type=float,
                        help="test nms threshold.")

    parser.add_argument('--use_np_file_as_input',
                        default=False,
                        type=bool,
                        required=False,
                        help="whether use dumped numpy file as input.")

    opt = parser.parse_args()

    logger.remove()#删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
    handler_id = logger.add(sys.stderr, level="INFO")#添加一个可以修改控制的handler

    save_path = os.path.join(
        save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )

    os.makedirs(save_path, exist_ok=True)

    cfg = None
    if opt.network == "mobile0.25":
        cfg = cfg_mnet
    elif opt.network == "resnet50":
        cfg = cfg_re50

    retinaface = Retinaface_sophon(
        cfg = cfg,
        bmodel_file_path=opt.bmodel,
        tpu_id=opt.tpu_id,
        score_threshold=opt.conf,
        nms_threshold=opt.nms)

    frame = cv2.imread(opt.input, 0)

    logger.info("processing file: {}".format(opt.input))

    if frame is not None:  # is picture file

         # TODO: unable to draw landmarks on bm_image now

        decoder = sail.Decoder(opt.input, False, 0)

        input_bmimg = sail.BMImage()
        ret = decoder.read(retinaface.handle, input_bmimg)
        if ret:
            logger.error("decode error\n")
            exit(-1)

        print_info(input_bmimg)

        for i in range(100):

            result_image = retinaface.predict_bmimage(input_bmimg) # opt.use_np_file_as_input
        
        retinaface.bmcv.imwrite(os.path.join(save_path, "test_output.jpg"), result_image)

    else:  # is video file

        # TODO: unable to draw landmarks on bm_image now

        decoder = sail.Decoder(opt.input, True, 0)

        if decoder.is_opened():

            logger.info("create decoder success")
            input_bmimg = sail.BMImage()
            id = 0

            while True:
                
                logger.warning("this is a video file ...")

                ret = decoder.read(retinaface.handle, input_bmimg)
                if ret:
                    logger.error("decoder error")
                    break

                result_image = retinaface.predict_bmimage(input_bmimg)

                retinaface.bmcv.imwrite(os.path.join(save_path, str(id) + ".jpg"), result_image)

                id += 1

            logger.warning("stream end or decoder error")

        else:
            logger.error("failed to create decoder")

    print("===================================================")

    from utils.time_utils import TimeStamp

    TimeStamp().print()