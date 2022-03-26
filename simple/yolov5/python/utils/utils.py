import numpy as np
import time
# import torch
# import torchvision


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def nms(bboxs, scores, iou_thresh=0.5):
    keep_indices = []
    scores_indices = np.argsort(scores)[::-1]

    while len(scores_indices) > 0:
        top = scores_indices[0]
        keep_indices.append(top)
        tmp = []
        for i in scores_indices[1:]:
            if scores[i] == scores[top]:
                tmp.append(i)
            else:
                break
        if len(tmp) == 0:
            scores_indices = scores_indices[1:]
        else:
            ious = box_iou(bboxs[top], bboxs[tmp])
            target_indices = np.where(ious <= iou_thresh)[0]
            keep_indices += [tmp[j] for j in target_indices]
            scores_indices = scores_indices[len(tmp) + 1:]
    return np.array(keep_indices)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6), dtype="float32")] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0] + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            tmp = x[:, 5:] > conf_thres
            print(np.nonzero(tmp)[1].shape)
            i, j = np.nonzero(tmp)
            # i, j = (x[:, 5:] > conf_thres).nonzero().T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype("float")), 1)
            # print(i, j, x)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdims=True)
            x = np.concatenate((box, conf, j.astype("float")), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not np.isfinite(x).all():
        #     x = x[np.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # print(boxes)
        # print(scores)

        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        #     if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #         # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #         iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #         weights = iou * scores[None]  # box weights
        #         x[i, :4] = np.matmul(weights, x[:, :4]) / weights.sum(1, keepdim=True)
        #         # x[i, :4] = np.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #         if redundant:
        #             i = i[iou.sum(1) > 1]  # require redundancy
        #
        output[xi] = x[i]
    #     if (time.time() - t) > time_limit:
    #         print(f'WARNING: NMS time limit {time_limit}s exceeded')
    #         break  # time limit exceeded
    #
    return output


# def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
#     """Performs Non-Maximum Suppression (NMS) on inference results
#     Returns:
#          detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
#     """
#
#     nc = prediction.shape[2] - 5  # number of classes
#     xc = prediction[..., 4] > conf_thres  # candidates
#
#     # Settings
#     min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
#     max_det = 300  # maximum number of detections per image
#     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#     time_limit = 10.0  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
#     merge = False  # use merge-NMS
#
#     t = time.time()
#     output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence
#
#         # Cat apriori labels if autolabelling
#         if labels and len(labels[xi]):
#             l = labels[xi]
#             v = torch.zeros((len(l), nc + 5), device=x.device)
#             v[:, :4] = l[:, 1:5]  # box
#             v[:, 4] = 1.0  # conf
#             v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
#             x = torch.cat((x, v), 0)
#
#         # If none remain process next image
#         if not x.shape[0]:
#             continue
#
#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
#
#         # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#         box = xywh2xyxy(x[:, :4])
#
#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
#         else:  # best class only
#             conf, j = x[:, 5:].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
#
#         # Filter by class
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
#
#         # Apply finite constraint
#         # if not torch.isfinite(x).all():
#         #     x = x[torch.isfinite(x).all(1)]
#
#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         elif n > max_nms:  # excess boxes
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
#
#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]
#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#             iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#             weights = iou * scores[None]  # box weights
#             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#             if redundant:
#                 i = i[iou.sum(1) > 1]  # require redundancy
#
#         output[xi] = x[i]
#         if (time.time() - t) > time_limit:
#             print(f'WARNING: NMS time limit {time_limit}s exceeded')
#             break  # time limit exceeded
#
#     return output
