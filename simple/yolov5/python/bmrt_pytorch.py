import torch
import numpy as np
import cv2


class Detector(object):
    def __init__(self):
        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.confThreshold = 0.5
        self.nmsThreshold = 0.5
        self.objThreshold = 0.5
        self.ration = 1
        with open('../data/coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def predict(self, tensor):
        # blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
        input_tensor = torch.Tensor(tensor)
        model = torch.jit.load("../data/models/yolov5s.torchscript.pt")
        outs = model(input_tensor)

        z = []  # inference output

        for i in range(self.nl):
            bs, _, ny, nx, nc, = outs[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if self.grid[i].shape[2:4] != outs[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-outs[i]))  ### sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

    def preprocess(self, img, target_size=640):
        self.target_size = target_size
        h, w, c = img.shape
        # Calculate widht and height and paddings
        r_w = target_size / w
        r_h = target_size / h
        if r_h > r_w:
            tw = target_size
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = 0
            ty2 = target_size - th - ty1
            self.ration = r_w
        else:
            tw = int(r_h * w)
            th = target_size
            tx1 = 0#int((target_size - tw) / 2)
            tx2 = target_size - tw - tx1
            ty1 = ty2 = 0
            self.ration=r_h

        # Resize long
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        # pad
        padded_img = cv2.copyMakeBorder(
            img, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
        )
        # BGR => RGB
        resized_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        # to tensor
        image = resized_img.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, padded_img, (max(r_w, r_h), tx1, ty1)

    def postprocess(self, src_img, outs):
        # frameHeight = frame.shape[0]
        # frameWidth = frame.shape[1]
        # ratioh, ratiow = frameHeight / 640, frameWidth / 640
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
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

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            #scale to the origin image
            left = box[0]/self.ration
            top = box[1]/self.ration
            x2 = (box[2] + box[0])/self.ration
            y2 = (box[1] + box[3])/self.ration
            height = box[3]/self.ration
            frame = self.drawPred(src_img, classIds[i], confidences[i], round(left), round(top), round(x2), round(y2))
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        print("classid=%d, conf=%f, (%d,%d,%d,%d)"%(classId, conf, left, top, right, bottom))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame


def main():
    YOLOv5 = Detector()
    # srcimg = cv2.imread("data/images/bus.jpg")
    src_img = cv2.imread("../data/images/bus.jpg")

    img, padded_img, (ratio, tx1, ty1) = YOLOv5.preprocess(src_img, target_size=640)
    print("img.shape: {}".format(img.shape))
    #######################
    # dump input tensor
    #np.save("torch_input.npy", img)
    bmrt_data = np.fromfile("../data/images/input_ref_data.dat.bmrt", dtype=np.float32).reshape(1,3,640,640)
    bmrt_data = torch.from_numpy(bmrt_data)
    dets = YOLOv5.predict(bmrt_data)
    print(dets.shape)
    # dump output tensor
    #np.save("torch_output.npy", dets)
    plot_img = YOLOv5.postprocess(src_img, dets)
    print(plot_img.shape)
    cv2.imwrite("torch_bmrt.jpg", plot_img)


if __name__ == '__main__':
    main()
