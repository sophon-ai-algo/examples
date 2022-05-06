import cv2
import subprocess as sp

rtspUrl = 'rtsp://11.73.12.184:554/test1' #这里改成本地ip，端口号不变，文件夹自定义
rtmpUrl = 'rtmp://8.136.105.93:10035/app/test'
# 视频来源 地址需要替换自己的可识别文件地址
sourceUrl='rtsp://admin:hk123456@11.73.12.20'
camera = cv2.VideoCapture(sourceUrl) # 从文件读取视频

# 视频属性
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + 'x' + str(size[1])
fps = camera.get(cv2.CAP_PROP_FPS)  # 30p/self
fps = int(fps)
hz = int(1000.0 / fps)
print('size:'+ sizeStr + ' fps:' + str(fps) + ' hz:' + str(hz))

# 直播管道输出
# ffmpeg推送rtmp 重点 ： 通过管道 共享数据的方式
command = [
    'ffmpeg',
    # 're',#
    '-y', # 无需询问即可覆盖输出文件
    '-an',
    '-hwaccel', 'bmcodec',
    '-hwaccel_device', '0',
    '-f', 'rawvideo', # 强制输入或输出文件格式
    '-vcodec','rawvideo', # 设置视频编解码器。这是-codec:v的别名
    '-pix_fmt', 'bgr24', # 设置像素格式
    '-s', sizeStr, # 设置图像大小
    '-r', str(fps), # 设置帧率
    '-i', '-', # 输入
    '-is_dma_buffer','0',
    '-c:v', 'h264_bm',
    #'-pix_fmt', 'yuv420p',
    # '-preset', 'ultrafast',
    '-f', 'flv',# 强制输入或输出文件格式
    rtmpUrl]

# # ffmpeg推送rtsp 重点 ： 通过管道 共享数据的方式
# command = [
#     'ffmpeg',
#     # 're',#
#     '-y', # 无需询问即可覆盖输出文件
#     '-an',
#     '-hwaccel', 'bmcodec',
#     '-hwaccel_device', '0',
#     '-f', 'rawvideo', # 强制输入或输出文件格式
#     '-vcodec','rawvideo', # 设置视频编解码器。这是-codec:v的别名
#     '-pix_fmt', 'bgr24', # 设置像素格式
#     '-s', sizeStr, # 设置图像大小
#     '-r', str(fps), # 设置帧率
#     '-i', '-', # 输入
#     '-is_dma_buffer','0',
#     '-c:v', 'h264_bm',
#     #'-pix_fmt', 'yuv420p',
#     # '-preset', 'ultrafast',
#     '-f', 'rtsp',# 强制输入或输出文件格式
#     rtspUrl]

#管道特性配置
pipe = sp.Popen(command, stdin=sp.PIPE) #,shell=False
while (camera.isOpened()):
    ret, frame = camera.read() # 逐帧采集视频流
    if not ret:
        break
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV_I420)
    ############################图片输出
    # 结果帧处理 存入文件 / 推流 / ffmpeg 再处理
    pipe.stdin.write(frame.tobytes())  # 存入管道用于直播

camera.release()