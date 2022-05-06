import math
from collections import namedtuple
import numpy as np
import np_methods
import cv2

SSDParams = namedtuple('SSDParameters', [
    'img_shape', 'num_classes', 'no_annotation_label', 'feat_layers',
    'feat_shapes', 'anchor_size_bounds', 'anchor_sizes', 'anchor_ratios',
    'anchor_steps', 'anchor_offset', 'normalizations', 'prior_scaling'
])


class SSDNet():
    default_params = SSDParams(img_shape=(300, 300),
                               num_classes=21,
                               no_annotation_label=21,
                               feat_layers=[
                                   'block4', 'block7', 'block8', 'block9',
                                   'block10', 'block11'
                               ],
                               feat_shapes=[(38, 38), (19, 19), (10, 10),
                                            (5, 5), (3, 3), (1, 1)],
                               anchor_size_bounds=[0.15, 0.90],
                               anchor_sizes=[(21., 45.), (45., 99.),
                                             (99., 153.), (153., 207.),
                                             (207., 261.), (261., 315.)],
                               anchor_ratios=[[2, .5], [2, .5, 3, 1. / 3],
                                              [2, .5, 3, 1. / 3],
                                              [2, .5, 3, 1. / 3], [2, .5],
                                              [2, .5]],
                               anchor_steps=[8, 16, 32, 64, 100, 300],
                               anchor_offset=0.5,
                               normalizations=[20, -1, -1, -1, -1, -1],
                               prior_scaling=[0.1, 0.1, 0.2, 0.2])

    def __init__(self, ssd_net, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params
        self.net = ssd_net
        self.anchors = self.anchors()

    def process_image(self,
                      inputs,
                      select_threshold=0.5,
                      nms_threshold=.45,
                      net_shape=(300, 300)):
        # Run SSD network.
        rbbox_img = np.array([0., 0., 1., 1.])
        rimg = self.pre_processing(inputs)
        rimg = rimg.astype(np.float32)
        self.net.fill_blob_data({self.net.inputs[0]: rimg})
        self.net.forward()
        rpredictions, rlocalisations = self.ssd_output_blobs(self.net)
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions,
            rlocalisations,
            self.anchors,
            select_threshold=select_threshold,
            img_shape=net_shape,
            num_classes=21,
            decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses,
                                                            rscores,
                                                            rbboxes,
                                                            top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(
            rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes

    def anchors(self, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return self.ssd_anchors_all_layers(self.params.img_shape,
                                           self.params.feat_shapes,
                                           self.params.anchor_sizes,
                                           self.params.anchor_ratios,
                                           self.params.anchor_steps,
                                           self.params.anchor_offset,
                                           dtype)

    def ssd_anchor_one_layer(self,
                             img_shape,
                             feat_shape,
                             sizes,
                             ratios,
                             step,
                             offset=0.5,
                             dtype=np.float32):
        """Computer SSD default anchor boxes for one feature layer.

        Determine the relative position grid of the centers, and the relative
        width and height.

        Arguments:
          feat_shape: Feature shape, used for computing relative position grids;
          size: Absolute reference sizes;
          ratios: Ratios to use on these features;
          img_shape: Image shape, used for computing height, width relatively to the
            former;
          offset: Grid offset.

        Return:
          y, x, h, w: Relative x and y grids, and height and width.
        """
        # Compute the position grid: simple way.
        # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        # y = (y.astype(dtype) + offset) / feat_shape[0]
        # x = (x.astype(dtype) + offset) / feat_shape[1]
        # Weird SSD-Caffe computation using steps values...
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset) * step / img_shape[0]
        x = (x.astype(dtype) + offset) * step / img_shape[1]

        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        # Compute relative height and width.
        # Tries to follow the original implementation of SSD for the order.
        num_anchors = len(sizes) + len(ratios)
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)
        # Add first anchor boxes with ratio=1.
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        di = 1
        if len(sizes) > 1:
            h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
            w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
            di += 1
        for i, r in enumerate(ratios):
            h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
        return y, x, h, w

    def ssd_anchors_all_layers(self,
                               img_shape,
                               layers_shape,
                               anchor_sizes,
                               anchor_ratios,
                               anchor_steps,
                               offset=0.5,
                               dtype=np.float32):
        """
        Compute anchor boxes for all feature layers.
        """
        layers_anchors = []
        for i, s in enumerate(layers_shape):
            anchor_bboxes = self.ssd_anchor_one_layer(img_shape,
                                                      s,
                                                      anchor_sizes[i],
                                                      anchor_ratios[i],
                                                      anchor_steps[i],
                                                      offset=offset,
                                                      dtype=dtype)
            layers_anchors.append(anchor_bboxes)
        return layers_anchors

    def ssd_output_blobs(self, ssd_net):
        """
        get ssd network output blobs and reorder them
        """
        predictions_keys = [
            'ssd_300_vgg/softmax/Reshape_1', 'ssd_300_vgg/softmax_1/Reshape_1',
            'ssd_300_vgg/softmax_2/Reshape_1',
            'ssd_300_vgg/softmax_3/Reshape_1',
            'ssd_300_vgg/softmax_4/Reshape_1',
            'ssd_300_vgg/softmax_5/Reshape_1'
        ]
        localisations_keys = [
            'ssd_300_vgg/block4_box/Reshape', 'ssd_300_vgg/block7_box/Reshape',
            'ssd_300_vgg/block8_box/Reshape', 'ssd_300_vgg/block9_box/Reshape',
            'ssd_300_vgg/block10_box/Reshape',
            'ssd_300_vgg/block11_box/Reshape'
        ]
        rpredictions = [
            ssd_net.get_blob_data(key + '@out').copy()
            for key in predictions_keys
        ]
        rlocalisations = [
            ssd_net.get_blob_data(key + '@out').copy()
            for key in localisations_keys
        ]
        return rpredictions, rlocalisations

    def pre_processing(self, img):
        '''
        process RGB image
        '''
        _R_MEAN = 123.
        _G_MEAN = 117.
        _B_MEAN = 104.
        img = np.array(img, dtype=np.float32)
        img = img - np.array([_R_MEAN, _G_MEAN, _B_MEAN]).reshape([1, 1, 3])
        img = cv2.resize(img,
                         self.params.img_shape,
                         interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis=0)
        return img.transpose([0, 3, 1, 2])
