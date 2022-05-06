import cv2
import os
import time
from multiprocessing import Process
import base64
import numpy as np

from utils.redis_client import RedisClientInstance
from utils import logger

log = logger.get_logger(__file__)


# 摄像机流地址
# rtsp://admin:a12345678@192.168.3.113:554/Streaming/Channels/101

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "frame")

RTSP_URL = "rtsp://admin:a12345678@192.168.3.113:554/Streaming/Channels/101"

class PullFrameProcess(Process):

    def __init__(self):
        super(PullFrameProcess, self).__init__()
        self.url = RTSP_URL
        self.exit_flag = 0
        self.redis_client = None

    def __del__(self):
        pass

    def run(self):
        # redis连接
        self.store = RedisClientInstance.get_storage_instance()

        log.info("start pull rtsp")
        vc = cv2.VideoCapture(self.url)
        while self.exit_flag == 0:
            try:

                ret, frame = vc.read()
                _, image = cv2.imencode('.jpg', frame)
                image_id = os.urandom(4)
                self.store.single_set_string('image', np.array(image).tobytes())
                self.store.single_set_string('image_id', image_id)
                log.info("store image id {}".format(image_id))
            except Exception as e:
                log.error("read frame error {}".format(e))
            cv2.waitKey(1)
        vc.release()

    def stop(self):
        self.exit_flag = 1



