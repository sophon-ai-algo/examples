import numpy as np
import sophon.sail as sail

class PreProcess:
    def __init__(self, cfg, input_scale=None):
        self.cfg = cfg
        mean_bgr = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        std_bgr = np.array([57.38, 57.12, 58.40], dtype=np.float32)
        self.mean = mean_bgr[::-1]  # bmcv use mean_rgb after bgr2rgb
        self.std = std_bgr[::-1]    # bmcv use std_rgb after bgr2rgb
        self.input_scale = float(1.0) if input_scale is None else input_scale

        self.normalize = cfg['normalize']
        self.subtract_means = cfg['subtract_means']
        self.to_float = cfg['to_float']

        self.width = cfg['width']
        self.height = cfg['height']

    def __call__(self, img, handle, bmcv):
        """
        pre-processing
        Args:
            img: sail.BMImage
            handle:
            bmcv:

        Returns: sail.BMImage after pre-processing

        """
        resized_img_rgb = sail.BMImage(handle, self.height, self.width,
                                       sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        # resize and bgr2rgb
        bmcv.vpp_resize(img, resized_img_rgb, self.width, self.height)

        preprocessed_img = sail.BMImage(handle, self.height, self.width,
                                        sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_FLOAT32)

        if self.normalize:
            a = 1 / self.std
            b = - self.mean / self.std
            alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])
            bmcv.convert_to(resized_img_rgb, preprocessed_img, alpha_beta)
        elif self.subtract_means:
            a = (1, 1, 1)
            b = - self.std
            alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])
            bmcv.convert_to(resized_img_rgb, preprocessed_img, alpha_beta)
        elif self.to_float:
            a = 1 / self.std
            b = (0, 0, 0)
            alpha_beta = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])
            bmcv.convert_to(resized_img_rgb, preprocessed_img, alpha_beta)

        return preprocessed_img

    def infer_batch(self, img_list, handle, bmcv):
        """
        batch pre-processing
        Args:
            img_list: a list of sail.BMImage
            handle:
            bmcv:

        Returns: a list of sail.BMImage after pre-processing

        """
        preprocessed_img_list = []
        for img in img_list:
            preprocessed_img = self(img, handle, bmcv)
            preprocessed_img_list.append(preprocessed_img)
        return preprocessed_img_list







