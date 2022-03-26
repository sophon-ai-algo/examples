import argparse
import cv2
import numpy as np
import sophon.sail as sail
import os
import time

from utils.colors import _COLORS

# YOLOV5 3output
# input: x.1, [1, 3, 640, 640], float32, scale: 1
# output: 172, [1, 3, 80, 80, 85], float32, scale: 1
# output: 173, [1, 3, 40, 40, 85], float32, scale: 1
# output: 174, [1, 3, 20, 20, 85], float32, scale: 1

opt = None
save_path = os.path.join(os.path.dirname(
    __file__), "output", os.path.basename(__file__).split('.')[0])


class YOLOV5_Detector(object):
    def __init__(self, bmodel_path, tpu_id, class_names_path, confThreshold=0.1, nmsThreshold=0.5, objThreshold=0.1):
        # load bmodel
        self.net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.handle = self.net.get_handle()

        # get model info
        self.graph_name = self.net.get_graph_names()[0]

        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(
            self.graph_name, self.input_name)
        self.input_w = int(self.input_shape[-1])
        self.input_h = int(self.input_shape[-2])
        self.input_shapes = {self.input_name: self.input_shape}
        self.input_dtype = self.net.get_input_dtype(
            self.graph_name, self.input_name)

        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)

        self.input = sail.Tensor(
            self.handle, self.input_shape, self.input_dtype, False, False)
        self.input_tensors = {self.input_name: self.input}

        self.output_name_large = self.net.get_output_names(self.graph_name)[0]
        self.output_name_middle = self.net.get_output_names(self.graph_name)[1]
        self.output_name_small = self.net.get_output_names(self.graph_name)[2]

        self.output_shape_large = self.net.get_output_shape(
            self.graph_name, self.output_name_large)
        self.output_shape_middle = self.net.get_output_shape(
            self.graph_name, self.output_name_middle)
        self.output_shape_small = self.net.get_output_shape(
            self.graph_name, self.output_name_small)

        self.output_dtype_large = self.net.get_output_dtype(
            self.graph_name, self.output_name_large)
        self.output_dtype_middle = self.net.get_output_dtype(
            self.graph_name, self.output_name_middle)
        self.output_dtype_small = self.net.get_output_dtype(
            self.graph_name, self.output_name_small)

        self.output_scale_large  = self.net.get_output_scale(self.graph_name, self.output_name_large)
        self.output_scale_middle = self.net.get_output_scale(self.graph_name, self.output_name_middle)
        self.output_scale_small  = self.net.get_output_scale(self.graph_name, self.output_name_small)

        self.output_large = sail.Tensor(
            self.handle, self.output_shape_large, self.output_dtype_large, True, True)
        self.output_middle = sail.Tensor(
            self.handle, self.output_shape_middle, self.output_dtype_middle, True, True)
        self.output_small = sail.Tensor(
            self.handle, self.output_shape_small, self.output_dtype_small, True, True)

        self.output_tensors = {self.output_name_large: self.output_large,
                               self.output_name_middle: self.output_middle, self.output_name_small: self.output_small}

        is_fp32 = (self.input_dtype == sail.Dtype.BM_FLOAT32)
        # get handle to create input and output tensors
        self.bmcv = sail.Bmcv(self.handle)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

        # bgr normalization
        self.ab = [x * self.input_scale for x in [0.003921568627451, 0, 0.003921568627451, 0, 0.003921568627451, 0]]

        # generate anchor
        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62,
                                              45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(
            anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])

        # post-process threshold
        scalethreshold = 1.0 if is_fp32 else 0.9
        self.confThreshold = confThreshold * scalethreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold * scalethreshold
        self.ration = 1

        with open(class_names_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        print("input img_dtype:{}, input scale: {}, output scale: {}, {}, {} ".format(
            self.img_dtype, self.input_scale, self.output_scale_large, self.output_scale_middle, self.output_scale_small))

    def compute_IOU(self, rec1, rec2):
        """
        计算两个矩形框的交并比。
        :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
        :param rec2: (x0,y0,x1,y1)
        :return: 交并比IOU.
        """
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        # 两矩形无相交区域的情况
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        # 两矩形有相交区域的情况
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * \
                (right_column_min - left_column_max)
            return S_cross / S1

    def sigmoid(self, inx):  # 防止exp(-x)溢出
        indices_pos = np.nonzero(inx >= 0)
        indices_neg = np.nonzero(inx < 0)

        y = np.zeros_like(inx)
        y[indices_pos] = 1 / (1 + np.exp(-inx[indices_pos]))
        y[indices_neg] = np.exp(inx[indices_neg]) / \
            (1 + np.exp(inx[indices_neg]))

        return y

    def preprocess_with_bmcv(self, img):

        img_w = img.width()
        img_h = img.height()
        
        # Calculate widht and height and paddings
        r_w = self.input_w / img_w
        r_h = self.input_h / img_h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = 0
            ty2 = self.input_h - th - ty1
            self.ration = r_w
        else:
            tw = int(r_h * img_w)
            th = self.input_h
            tx1 = 0
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
            self.ration = r_h

        attr = sail.PaddingAtrr()
        attr.set_stx(tx1)
        attr.set_sty(ty1)
        attr.set_w(tw)
        attr.set_h(th)
        attr.set_r(114)
        attr.set_g(114)
        attr.set_b(114)

        print("original image format:{}".format(img.format()))

        # preprocess 
        padded_img_bgr = self.bmcv.vpp_resize_padding(
            img, self.input_shape[2], self.input_shape[3], attr)
        self.bmcv.imwrite(os.path.join(
            save_path, "padded_img_bgr.bmp"), padded_img_bgr)
        print("image format after vpp_resize_padding:{}".format(padded_img_bgr.format()))

        padded_img_rgb = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3],
                                      sail.Format.FORMAT_RGB_PLANAR, padded_img_bgr.dtype())
        self.bmcv.vpp_resize(padded_img_bgr, padded_img_rgb,
                             self.input_shape[2], self.input_shape[3])
        print("image format after vpp_resize:{}".format(padded_img_rgb.format()))

        self.bmcv.imwrite(os.path.join(
            save_path, "padded_img_rgb.bmp"), padded_img_rgb)

        padded_img_rgb_norm = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3],
                                           sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(
            padded_img_rgb, padded_img_rgb_norm, ((self.ab[0], self.ab[1]), \
                                        (self.ab[2], self.ab[3]), \
                                        (self.ab[4], self.ab[5])))

        print("image format after convert_to:{}".format(padded_img_rgb_norm.format()))

        # img_back = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3], \
        #                 sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        # a = 255
        # b = 0
        # self.bmcv.convert_to(img_rgb, img_back, ((a, b), (a, b), (a, b)))
        # print("image format after convert_to:{}".format(img_back.format()))
        # self.bmcv.imwrite("conver_1.bmp", img_back)
        
        return padded_img_rgb_norm, (min(r_w, r_h), tx1, ty1)

    def preprocess_with_bmcv_center(self, img):

        img_w = img.width()
        img_h = img.height()

        # Calculate widht and height and paddings
        r_w = self.input_w / img_w
        r_h = self.input_h / img_h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
            self.ration = r_w
        else:
            tw = int(r_h * img_w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
            self.ration = r_h

        attr = sail.PaddingAtrr()
        attr.set_stx(tx1)
        attr.set_sty(ty1)
        attr.set_w(tw)
        attr.set_h(th)
        attr.set_r(0)
        attr.set_g(0)
        attr.set_b(0)

        print("original image format:{}".format(img.format()))

        padded_img_bgr = self.bmcv.vpp_resize_padding(
            img, self.input_shape[2], self.input_shape[3], attr)
        self.bmcv.imwrite(os.path.join(
            save_path, "padded_img_bgr_center.bmp"), padded_img_bgr)
        print("image format after vpp_resize_padding:{}".format(padded_img_bgr.format()))

        padded_img_rgb = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3],
                                      sail.Format.FORMAT_RGB_PLANAR, padded_img_bgr.dtype())
        self.bmcv.vpp_resize(padded_img_bgr, padded_img_rgb,
                             self.input_shape[2], self.input_shape[3])
        print("image format after vpp_resize:{}".format(padded_img_rgb.format()))
        self.bmcv.imwrite(os.path.join(
            save_path, "padded_img_rgb_center.bmp"), padded_img_rgb)

        padded_img_rgb_norm = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3],
                                           sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(
            padded_img_rgb, padded_img_rgb_norm, ((self.ab[0], self.ab[1]), \
                                        (self.ab[2], self.ab[3]), \
                                        (self.ab[4], self.ab[5])))
        print("image format after convert_to:{}".format(padded_img_rgb_norm.format()))

        # img_back = sail.BMImage(self.handle, self.input_shape[2], self.input_shape[3], \
        #                 sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        # a = 255
        # b = 0
        # self.bmcv.convert_to(img_rgb, img_back, ((a, b), (a, b), (a, b)))
        # print("image format after convert_to:{}".format(img_back.format()))
        # self.bmcv.imwrite("conver_1.bmp", img_back)

        return padded_img_rgb_norm, (min(r_w, r_h), tx1, ty1)


    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float")

    def predict(self, data, use_np_file_as_input):

        z = []  # inference output

        if use_np_file_as_input:
            print("use numpy data as input")
            ref_data = np.load("./np_input.npy")
            print(ref_data.shape)

            input = sail.Tensor(self.handle, ref_data)
            input_tensors = {self.input_name: input}
            input_shapes = {self.input_name: self.input_shape}

            self.net.process(self.graph_name, input_tensors,
                             input_shapes, self.output_tensors)
        else:
            print("use decoder data as input")
            self.bmcv.bm_image_to_tensor(data, self.input)
            self.net.process(self.graph_name, self.input_tensors,
                             self.input_shapes, self.output_tensors)

        output_large = self.output_large.asnumpy(self.output_shape_large) * self.output_scale_large
        output_middle = self.output_middle.asnumpy(self.output_shape_middle) * self.output_scale_middle
        output_small = self.output_small.asnumpy(self.output_shape_small) * self.output_scale_small

        outputs_list = [output_large, output_middle, output_small]

        for i, feat in enumerate(outputs_list):
            np.save("tensor_output_"+str(i), feat)
            # feat = np.load("np_"+str(i)+".npy")
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx, nc = feat.shape
            if self.grid[i].shape[2:4] != feat.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = self.sigmoid(feat)
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                           self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

    def predict_center(self, data, use_np_file_as_input):

        z = []  # inference output

        if use_np_file_as_input:
            print("use numpy data as input")
            ref_data = np.load("./np_input_center.npy")
            print(ref_data.shape)

            input = sail.Tensor(self.handle, ref_data)
            input_tensors = {self.input_name: input}
            input_shapes = {self.input_name: self.input_shape}

            self.net.process(self.graph_name, input_tensors,
                             input_shapes, self.output_tensors)
        else:
            print("use decoder data as input")
            self.bmcv.bm_image_to_tensor(data, self.input)
            self.net.process(self.graph_name, self.input_tensors,
                             self.input_shapes, self.output_tensors)

        # output_large = self.output_large.asnumpy(self.output_shape_large)
        # output_middle = self.output_middle.asnumpy(self.output_shape_middle)
        # output_small = self.output_small.asnumpy(self.output_shape_small)

        output_large = self.output_large.asnumpy(self.output_shape_large) * self.output_scale_large
        output_middle = self.output_middle.asnumpy(self.output_shape_middle) * self.output_scale_middle
        output_small = self.output_small.asnumpy(self.output_shape_small) * self.output_scale_small

        outputs_list = [output_large, output_middle, output_small]

        for output in outputs_list:
            print("output shape : {}".format(output.shape))

        for i, feat in enumerate(outputs_list):
            np.save("tensor_output_"+str(i), feat)
            # feat = np.load("np_"+str(i)+".npy")
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx, nc = feat.shape
            if self.grid[i].shape[2:4] != feat.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = self.sigmoid(feat)
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                           self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

    def postprocess(self, outs):
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            out = out[out[:, 4] > self.objThreshold, :]
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0])
                    center_y = int(detection[1])
                    width = int(detection[2])
                    height = int(detection[3])

                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence)*detection[4])
                    boxes.append([left, top, width, height])
        # Perform nms to eliminate redundant overlapping boxes with lower confidences.
        print("boxes.len: {}, confidences.len: {}".format(len(boxes), len(confidences)))
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold)
        return indices, boxes, confidences, classIds

    def inference(self, frame, use_np_file_as_input=False):

        if not isinstance(frame, type(None)):

            img, (ratio, tx1, ty1) = self.preprocess_with_bmcv(frame)

            dets = self.predict(img, use_np_file_as_input)
            print(dets.shape)

            indices, boxes, confidences, classIds = self.postprocess(dets)

            res = []
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = int((box[0]) / ratio)
                top = int((box[1]) / ratio)
                right = int((box[2] + box[0])/ratio)
                bottom = int((box[1] + box[3])/ratio)
                width = right - left
                height = bottom - top

                res.append({'det_box': [left, top, width, height],
                           "conf": confidences[i], "classId": classIds[i]})

                result_image = self.drawPred(frame, classIds[i], confidences[i], round(
                    left), round(top), round(right), round(bottom))
            return result_image

    def inference_center(self, frame, use_np_file_as_input=False):

        if not isinstance(frame, type(None)):

            img, (ratio, tx1, ty1) = self.preprocess_with_bmcv_center(frame)

            print("input shape : ", (img.width(), img.height()))

            dets = self.predict_center(img, use_np_file_as_input)

            indices, boxes, confidences, classIds = self.postprocess(dets)

            res = []
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = int((box[0] - tx1) / ratio)
                top = int((box[1] - ty1) / ratio)
                width = int(box[2] / ratio)
                height = int(box[3] / ratio)
                right = left + width
                bottom = top + height

                res.append({'det_box': [left, top, width, height],
                           "conf": confidences[i], "classId": classIds[i]})

                result_image = self.drawPred(frame, classIds[i], confidences[i], round(
                    left), round(top), round(right), round(bottom))

            return result_image

    def drawPred(self, frame, classId, conf, left, top, right, bottom):

        color = (_COLORS[classId] * 255).astype(np.uint8).tolist()

        print("classid=%d, class=%s, conf=%f, (%d,%d,%d,%d)" %
              (classId, self.classes[classId], conf, left, top, right, bottom))
              
        # draw bboxes
        self.bmcv.rectangle(frame, int(left), int(top), int(right-left), int(bottom-top), color, 3)

        # label = '%.2f' % conf
        # label = '%s:%s' % (self.classes[classId], label)
        # # Display the label at the top of the bounding box
        # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[classId], thickness=2)
        return frame


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Demo of YOLOv5 with preprocess by BMCV")

    parser.add_argument('--bmodel',
                        type=str,
                        default="../data/models/yolov5s_640_coco_v6.1_3output_int8_1b.bmodel",
                        required=False,
                        help='bmodel file path.')

    parser.add_argument('--labels',
                        type=str,
                        default="../data/coco.names",
                        required=False,
                        help='labels txt file path.')

    parser.add_argument('--input',
                        type=str,
                        default="../data/images/bus.jpg",
                        required=False,
                        help='input pic/video file path.')

    parser.add_argument('--tpu_id',
                        default=0,
                        type=int,
                        required=False,
                        help='tpu dev id(0,1,2,...).')

    parser.add_argument("--conf",
                        default=0.1,
                        type=float,
                        help="test conf threshold.")

    parser.add_argument("--nms",
                        default=0.5,
                        type=float,
                        help="test nms threshold.")

    parser.add_argument("--obj",
                        default=0.1,
                        type=float,
                        help="test obj conf.")

    parser.add_argument('--use_np_file_as_input',
                        default=False,
                        type=bool,
                        required=False,
                        help="whether use dumped numpy file as input.")

    opt = parser.parse_args()

    save_path = os.path.join(
        save_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    )

    if opt.use_np_file_as_input:
        save_path = save_path + "_numpy"

    os.makedirs(save_path, exist_ok=True)

    yolov5 = YOLOV5_Detector(bmodel_path=opt.bmodel,
                             tpu_id=opt.tpu_id,
                             class_names_path=opt.labels,
                             confThreshold=opt.conf,
                             nmsThreshold=opt.nms,
                             objThreshold=opt.obj)

    frame = cv2.imread(opt.input)

    print("processing file: {}".format(opt.input))

    if frame is not None:  # is picture file

        # decode
        decoder = sail.Decoder(opt.input, True, 0)

        input_bmimg = sail.BMImage()
        ret = decoder.read(yolov5.handle, input_bmimg)
        if ret:
            print("decode error\n")
            exit(-1)

        result_image = yolov5.inference_center(input_bmimg, opt.use_np_file_as_input)

        yolov5.bmcv.imwrite(os.path.join(save_path, "test_output.jpg"), result_image)

    else:  # is video file

        decoder = sail.Decoder(opt.input, True, 0)

        if decoder.is_opened():

            print("create decoder success")
            input_bmimg = sail.BMImage()
            id = 0

            while True:
                
                print("123")

                ret = decoder.read(yolov5.handle, input_bmimg)
                if ret:
                    print("decoder error\n")
                    break

                result_image = yolov5.inference_center(input_bmimg, use_np_file_as_input=False)

                yolov5.bmcv.imwrite(os.path.join(save_path, str(id) + ".jpg"), result_image)

                id += 1

            print("stream end or decoder error")

        else:
            print("failed to create decoder")

    print("===================================================")
