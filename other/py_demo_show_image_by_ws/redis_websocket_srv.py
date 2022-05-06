import json
from multiprocessing import Process
import time
import tornado
import asyncio
import base64

from tornado import ioloop, web, websocket, httpclient
from tornado.web import RequestHandler

from utils.redis_client import RedisClientInstance
from utils import logger

log = logger.get_logger(__file__)


MAX_FPS = 100

class IndexHandler(web.RequestHandler):
    """ Handler for the root static page. """

    def get(self):
        """ Retrieve the page content. """
        self.render('index.html')


# WebsocketHandller().连接建立时，将callback注册到register，
# 连接关闭时清理自己的callback。
class ImageWebSocketHandler(websocket.WebSocketHandler):

    def __init__(self, *args, **kwargs):
        super(ImageWebSocketHandler, self).__init__(*args, **kwargs)

        self._store  = RedisClientInstance.get_storage_instance()
        self._prev_image_id = None

    def open(self):
        log.info("加入新客户端, time={}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    def on_message(self, message):
        log.info("message {}".format(message))
        # 收到消息后发送信息
        while True:
            time.sleep(1. / MAX_FPS)
            image_id = self._store.get_by_name('image_id')
            if image_id != self._prev_image_id:
                break
        self._prev_image_id = image_id
        image = self._store.get_by_name('image')
        image = base64.b64encode(image)
        self.write_message(image)

    def on_close(self):
        log.info("关闭客户端, time={}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    def check_origin(self, origin):
        return True

    def on_ping(self, data):
        """ 心跳包响应, data是`.ping`发出的数据 """

        log.info('into on_pong the data is |%s|' % data)


# websocket服务线程
class WebsocketProcess(Process):
    def __init__(self):
        super(WebsocketProcess, self).__init__()
        self.exit_flag = 0

    def __del__(self):
        pass

    def stop(self):
        self.exit_flag = 1

    def run(self):
        log.info('start run app')
        # 线程服务
        # asyncio.set_event_loop(asyncio.new_event_loop())
        # 启动服务
        app = web.Application([
            (r'/', IndexHandler),
            (r'/ws', ImageWebSocketHandler),
        ])
        app.listen(8000)
        ioloop.IOLoop.instance().start()
