#include <string.h>
#include <sstream>
#include <queue>
#include <vector>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <sys/syscall.h>
#include <libavformat/avformat.h>
#include "yolox.hpp"

struct DataQueue
{
    int id;
    bm_image* bmimage;
    uint64_t num;
    std::vector<ObjRect> rect_out;
};
