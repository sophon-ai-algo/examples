import os
import cv2
import numpy as np
import argparse
import sophon.sail as sail
from utils.utils import xywh2xyxy, nms_np
from utils.colors import _COLORS

# YOLOV5 1 output
# input: x.1, [1, 3, 640, 640], float32, scale: 1
# output: 170, [1, 25200, 85], float32, scale: 1

class Detector(object):
    def __init__(self, img_size, tpu_id=0, model_format="fp32"):
        # load bmodel
        model_path = os.path.join(os.path.dirname(__file__),
                                  "../data/models/yolov5s_{}_{}_1.bmodel".format(model_format, img_size))
        print("using model {}".format(model_path))
        self.net = sail.Engine(model_path, tpu_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        # generate anchor
        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        # post-process threshold
        self.confThreshold = 0.5
        self.nmsThreshold = 0.5
        self.objThreshold = 0.5
        self.img_size = img_size
        coco_path = os.path.join(os.path.dirname(__file__),
                                  "../data/coco.names")
        with open(coco_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def preprocess(self, img):
        target_size = self.img_size
        h, w, c = img.shape
        # Calculate widht and height and paddings
        r_w = target_size / w
        r_h = target_size / h
        if r_h > r_w:
            tw = target_size
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((target_size - th) / 2)
            ty2 = target_size - th - ty1
        else:
            tw = int(r_h * w)
            th = target_size
            tx1 = int((target_size - tw) / 2)
            tx2 = target_size - tw - tx1
            ty1 = ty2 = 0
        # Resize long
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        # pad
        padded_img = cv2.copyMakeBorder(
            img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
        )
        # BGR => RGB
        resized_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("resized.jpg", resized_img)
        # to tensor
        #image = resized_img.astype(self.input_dtype)
        image = resized_img.astype(np.float32)
        image /= 255.0

        print(image.max(), image.min())
        # Normalize to [0,1]
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, padded_img, (min(r_w, r_h), tx1, ty1)

    def predict(self, tensor):
        input_data = {self.input_name: np.array(tensor, dtype=np.float32)}
        output = self.net.process(self.graph_name, input_data)
        return list(output.values())[0]

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float")

    def postprocess_np(self, outs, max_wh=7680):
        bs = outs.shape[0]
        output = [np.zeros((0, 6))] * bs
        xc = outs[..., 4] > self.confThreshold
        for xi, x in enumerate(outs):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            conf = x[:, 5:].max(1)
            j = x[:, 5:].argmax(1)
            x = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1)), 1)[conf > self.confThreshold]
            c = x[:, 5:6] * max_wh  # classes
            boxes = x[:, :4] + c.reshape(-1, 1)
            scores = x[:, 4]
            i = nms_np(boxes, scores, self.nmsThreshold)
            output[xi] = x[i]

        return output

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


def main(opt):
    src_img = cv2.imread(opt.img_name)
    if src_img is None:
        print("Error: reading image '{}'".format(img_name))
        return -1

    YOLOv5 = Detector(opt.img_size, tpu_id=opt.tpu, model_format = opt.format)

    img, padded_img, (ratio, tx1, ty1) = YOLOv5.preprocess(src_img)
    print("img.shape: {}".format(img.shape))
    
    dets = YOLOv5.predict(img)
    print(dets.shape)

    output = YOLOv5.postprocess_np(dets)

    result_image = src_img
    for det in output[0]:
        # label = self.classes[det[5]]
        box = det[:4]

        # scale to the origin image
        left = int((box[0] - tx1) / ratio)
        top = int((box[1] - ty1) / ratio)
        right = int((box[2] - tx1) / ratio)
        bottom = int((box[3] - ty1) / ratio)

        result_image = YOLOv5.drawPred(result_image, int(det[5]), det[4], round(
            left), round(top), round(right), round(bottom))
    print(result_image.shape)
    cv2.imwrite(opt.out_name, result_image)

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    image_path = os.path.join(os.path.dirname(__file__),"../data/images/zidane.jpg")
    parser.add_argument('--img-name', type=str, default=image_path, help='input image name')
    parser.add_argument('--out-name', type=str, default="sail.jpg", help='output image name')
    parser.add_argument('--tpu', type=int, default=0, help='tpu id')
    parser.add_argument('--format', type=str, default="fp32", help='model format fp32 or fix8b')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
