import cv2
import os
import time
import signal
import multiprocessing as mp

from utils import logger
from pull_frame import PullFrameProcess
from redis_websocket_srv import WebsocketProcess

log = logger.get_logger(__file__)

'''
测试websocket发送服务
1. opencv 解码视频流，帧信息存入redis
2. 从redis中取出帧信息， 通过websocket发送给前端
3. 前端接收websocket， 展示在页面上
'''


# 任务列表
service_list = []

def service_start():
    for service in service_list:
        service.start()

def service_join():
    for service in service_list:
        service.join()


def exit_handler(signum, frame):
    os._exit(0)


def main():
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)

    pull_srv = PullFrameProcess()
    pull_srv.daemon = True
    service_list.append(pull_srv)

    web_srv = WebsocketProcess()
    web_srv.daemon = True
    service_list.append(web_srv)

    # 任务开启
    service_start()
    service_join()


if __name__ == "__main__":
    main()
    exit(0)
