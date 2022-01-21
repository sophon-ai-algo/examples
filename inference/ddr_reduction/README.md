# DDR Reduction 介绍
目前AI结构化应用中，为了跟踪和择优，往往需要保存一定数量的视频数据帧，此数据帧格式一般为YUV/RGB等，这些裸数据占用的空间一般都比较大，以YUV为例：1080P RGB格式的一帧数据的大小为：1920*1080*3=6220800 bytes 约等于6.2M bytes, 如果缓存50帧的话，需要占用的空间为：311M。  

如果是16路视频同时分析，缓存的视频帧占用空间大约为：311M * 16 = 4976M 大约5G内存，这个内存是非常庞大的。

为了在结构话应用中减少内存占用，提出缓存的时候，不要缓存解码后的帧，而是缓存解码前的帧，这样就会大大减少内存的占用。

众说周知，H264/H265的压缩率很高，可以几十倍，甚至上百倍的压缩，为了实现此方法，算能提供了一个参考实现。

## 接口介绍
``` C++
class DDRReduction {
public:
    //
    //创建模块
    //@dev_id: Sophon设备ID
    //@codecId: 要创建的解码器类型：目前支持h264/hevc.
    // 成功会返回一个模块的智能指针
    //
    static std::shared_ptr<DDRReduction> create(int dev_id, AVCodecID codecId);

    //
    //析构函数
    //
    virtual ~DDRReduction() {}

    //
    //输入一个数据流帧，并返回一个唯一ID，通过此ID可以获取对应的视频帧
    //@pkt: 网络收到的数据包，AVPacket格式。
    //@p_id:返回的数据包Id， 可以为空
    //
    virtual int put_packet(AVPacket *pkt, int64_t *p_id) = 0;

    //
    //输入一个数据流帧，然后解码通过函数回调通知
    //@cb： 用户需要自己实现这个回调函数，规划pkt_id和frame的使用方法。
    //@pkt: 网络收到的数据包，AVPacket格式。
    //
    virtual int put_packet(AVPacket *pkt, std::function<void(int64_t pkt_id, AVFrame* frame)> cb) = 0;

    //
    //随机根据参考数据包ID，获取下一包对应的视频帧和pkt_id
    //@reference_id： 参考包ID， 不知道可以填-1， 这样指定找下一个最接近的包。
    //@frame：如果正确，返回的视频帧
    //@got_frame: 如果正确，返回是否得到视频帧
    //@p_id： 返回视频帧对应的包ID
    //
    virtual int seek_frame(int64_t reference_id, AVFrame *frame, int *got_frame, int64_t *p_id) = 0;

    //
    // 释放对应的包
    //
    virtual int free_packet(int64_t id) = 0;

    //
    // 清空所有包缓存。
    //
    virtual int flush() = 0;

    //
    // 获取统计信息
    //
    virtual int get_stat(DDRReductionStat *stat) = 0;
};
```

## API用法

请参考
* ddr_reduction_unittest.cpp
* cvs10