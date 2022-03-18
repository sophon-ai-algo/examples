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
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(
            self.graph_name, self.input_name)
        self.input_w = int(self.input_shape[-1])
        self.input_h = int(self.input_shape[-2])

        # generate anchor
        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62,
                                              45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(
            anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])

        # post-process threshold
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.ratio = 1

        with open(class_names_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def preprocess(self, img, c, h, w):

        img_h, img_w, img_c = img.shape

        # Calculate widht and height and paddings
        r_w = w / img_w
        r_h = h / img_h
        if r_h > r_w:
            tw = w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = 0  # int((h - th) / 2)
            ty2 = h - th - ty1
            self.ratio = r_w
        else:
            tw = int(r_h * img_w)
            th = h
            tx1 = 0  # int((w - tw) / 2)
            tx2 = w - tw - tx1
            ty1 = ty2 = 0
            self.ratio = r_h

        # Resize long
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        # pad
        padded_img_bgr = cv2.copyMakeBorder(
            img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
        )
        cv2.imwrite(os.path.join(
            save_path, "padded_img_bgr.jpg"), padded_img_bgr)
        # BGR => RGB
        padded_img_rgb = cv2.cvtColor(padded_img_bgr, cv2.COLOR_BGR2RGB)
        # to tensor
        padded_img_rgb_data = padded_img_rgb.astype(np.float32)
        # Normalize to [0,1]
        padded_img_rgb_data /= 255.0
        # HWC to CHW format:
        padded_img_rgb_data = np.transpose(padded_img_rgb_data, [2, 0, 1])
        # CHW to NCHW format
        padded_img_rgb_data = np.expand_dims(padded_img_rgb_data, axis=0)
        # Convert the image to row-major order, also known as "C order":
        padded_img_rgb_data = np.ascontiguousarray(padded_img_rgb_data)
        np.save('np_input', padded_img_rgb_data)
        return padded_img_rgb_data, (min(r_w, r_h), tx1, ty1)

    def preprocess_center(self, img, c, h, w):

        img_h, img_w, img_c = img.shape

        # Calculate widht and height and paddings
        r_w = w / img_w
        r_h = h / img_h
        if r_h > r_w:
            tw = w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = int((h - th) / 2)
            ty2 = h - th - ty1
            self.ration = r_w
        else:
            tw = int(r_h * img_w)
            th = h
            tx1 = int((w - tw) / 2)
            tx2 = w - tw - tx1
            ty1 = ty2 = 0
            self.ration = r_h

        # Resize long
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        # pad
        padded_img_bgr = cv2.copyMakeBorder(
            img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
        )
        cv2.imwrite(os.path.join(
            save_path, "padded_img_bgr_center.jpg"), padded_img_bgr)
        # BGR => RGB
        padded_img_rgb = cv2.cvtColor(padded_img_bgr, cv2.COLOR_BGR2RGB)
        # to tensor
        padded_img_rgb_data = padded_img_rgb.astype(np.float32)
        # Normalize to [0,1]
        padded_img_rgb_data /= 255.0
        # HWC to CHW format:
        padded_img_rgb_data = np.transpose(padded_img_rgb_data, [2, 0, 1])
        # CHW to NCHW format
        padded_img_rgb_data = np.expand_dims(padded_img_rgb_data, axis=0)
        # Convert the image to row-major order, also known as "C order":
        padded_img_rgb_data = np.ascontiguousarray(padded_img_rgb_data)
        np.save('np_input_center', padded_img_rgb_data)
        return padded_img_rgb_data, (min(r_w, r_h), tx1, ty1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float")

    def predict(self, input_data, use_np_file_as_input):
        # blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
        #ref_data = np.fromfile("./input_ref_data.dat.bmrt", dtype=np.float32)
        #ref_data = np.reshape(ref_data, newshape=(1,3,640,-1))
        #input_data = {self.input_name: ref_data}
        if use_np_file_as_input:
            print("use numpy data as input")
            input_data = np.load("./np_input.npy")
        else:
            print("use decode data as input")

        input = {self.input_name: np.array(input_data, dtype=np.float32)}
        output = self.net.process(self.graph_name, input)

        z = []  # inference output

        sorted_list = sorted(output.items(), key=lambda x: x[0])
        for i, (key, feat) in enumerate(sorted_list):
            np.save("np_"+str(i), feat)
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx, nc = feat.shape
            if self.grid[i].shape[2:4] != feat.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-feat))  # sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                           self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

    def predict_center(self, input_data, use_np_file_as_input):
        # blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
        #ref_data = np.fromfile("./input_ref_data.dat.bmrt", dtype=np.float32)
        #ref_data = np.reshape(ref_data, newshape=(1,3,640,-1))
        #input_data = {self.input_name: ref_data}
        if use_np_file_as_input:
            print("use numpy data as input")
            input_data = np.load("./np_input_center.npy")
        else:
            print("use decode data as input")

        print("input_data shape: ", input_data.shape)

        input = {self.input_name: np.array(input_data, dtype=np.float32)}
        output = self.net.process(self.graph_name, input)

        z = []  # inference output

        sorted_list = sorted(output.items(), key=lambda x: x[0])
        for i, (key, feat) in enumerate(sorted_list):
            np.save("np_"+str(i), feat)
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx, nc = feat.shape
            if self.grid[i].shape[2:4] != feat.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-feat))  # sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                           self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

    def postprocess(self, outs):
        # frameHeight = frame.shape[0]
        # frameWidth = frame.shape[1]
        # ratioh, ratiow = frameHeight / 640, frameWidth / 640

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        print("outs.shape: {}".format(outs.shape))
        for out in outs:
            out = out[out[:, 4] > self.objThreshold, :]
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold and detection[4] > self.objThreshold:
                    # center_x = int(detection[0] * ratiow)
                    # center_y = int(detection[1] * ratioh)
                    # width = int(detection[2] * ratiow)
                    # height = int(detection[3] * ratioh)
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
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold)

        return indices, boxes, confidences, classIds

    def inference(self, frame, use_np_file_as_input=False):
        input_data, (ratio, tx1, ty1) = self.preprocess(
            frame, 3, self.input_h, self.input_w)
        print("input_data.shape: {}".format(input_data.shape))

        dets = self.predict(input_data, use_np_file_as_input)
        print(dets.shape)

        indices, boxes, confidences, classIds = self.postprocess(dets)

        for i in indices:
            i = i[0]
            box = boxes[i]
            # scale to the origin image
            left = box[0]/ratio
            top = box[1]/ratio
            right = (box[2] + box[0])/ratio
            bottom = (box[1] + box[3])/ratio

            result_image = self.drawPred(frame, classIds[i], confidences[i], round(
                left), round(top), round(right), round(bottom))

        return result_image

    def inference_center(self, frame, use_np_file_as_input=False):
        input_data, (ratio, tx1, ty1) = self.preprocess_center(
            frame, 3, self.input_h, self.input_w)
        print("input_data.shape: {}".format(input_data.shape))

        dets = self.predict_center(input_data, use_np_file_as_input)
        print(dets.shape)

        indices, boxes, confidences, classIds = self.postprocess(dets)

        for i in indices:
            i = i[0]
            box = boxes[i]
            # scale to the origin image
            left = int((box[0] - tx1) / ratio)
            top = int((box[1] - ty1) / ratio)
            width = int(box[2] / ratio)
            height = int(box[3] / ratio)
            right = left + width
            bottom = top + height

            result_image = self.drawPred(frame, classIds[i], confidences[i], round(
                left), round(top), round(right), round(bottom))

        return result_image

    def drawPred(self, frame, classId, conf, left, top, right, bottom):

        color = (_COLORS[classId] * 255).astype(np.uint8).tolist()

        print("classid=%d, class=%s, conf=%f, (%d,%d,%d,%d)" %
              (classId, self.classes[classId], conf, left, top, right, bottom))

        cv2.rectangle(frame, (left, top), (right, bottom),
                      color, thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        return frame


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Demo of YOLOv5 with preprocess by OpenCV")

    parser.add_argument('--bmodel',
                        type=str,
                        default="../data/models/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel",
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

        result_image = yolov5.inference(frame, opt.use_np_file_as_input)

        print(result_image.shape)

        cv2.imwrite(os.path.join(
            save_path, "test_output.jpg"), result_image)

    else:  # is video file

        cap = cv2.VideoCapture(opt.input)

        ret, frame = cap.read()

        id = 0

        while ret:

            result_image = yolov5.inference(
                frame, use_np_file_as_input=False)

            print(result_image.shape)
            cv2.imwrite(os.path.join(save_path, save_path +
                        str(id) + ".jpg"), result_image)

            id += 1

            ret, frame = cap.read()

        print("stream end or decoder error")

        cap.release()
    
    print("===================================================")
