//
// Created by hsyuan on 2019-03-20.
//

#include "video_opengl_widget.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QDebug>
#include <QPainter>
#include <QTimer>

#define VERTEXIN 0
#define TEXTUREIN 1
#define RECT_DELAY_TIME 3

video_opengl_widget::video_opengl_widget(QWidget *parent): m_start_playing(false), m_last_frame(0), m_last_draw_time_us(0),
        QOpenGLWidget(parent)
{
    m_refreshTimer = new QTimer(this);
    m_refreshTimer->setTimerType(Qt::PreciseTimer);
    connect(m_refreshTimer, SIGNAL(timeout()), this, SLOT(onRefreshTimeout()));
    m_refreshTimer->setInterval(40);
    m_refreshTimer->start();
}

video_opengl_widget::~video_opengl_widget()
{
#if USE_CACHE
    while (m_cache_frames.size() > 0) {
        auto f = m_cache_frames.front();
        m_cache_frames.pop_front();
        av_frame_free(&f);
    }
#endif

    if (m_last_frame) {
        av_frame_free(&m_last_frame);
    }

    makeCurrent();

    m_yuv420p_vbo.destroy();
    if (m_yuv420p_textureY) m_yuv420p_textureY->destroy();
    if (m_yuv420p_textureU) m_yuv420p_textureU->destroy();
    if (m_yuv420p_textureV) m_yuv420p_textureV->destroy();

    m_nv12_vbo.destroy();
    if (m_nv12_textureY) m_nv12_textureY->destroy();
    if (m_nv12_textureUV) m_nv12_textureUV->destroy();

    doneCurrent();
}

void video_opengl_widget::slot_draw_frame(const AVFrame *frame)
{
    draw_frame(frame);
}

int video_opengl_widget::frame_height() {
    return videoH;
}

int video_opengl_widget::frame_width() {
    return videoW;
}

int video_opengl_widget::draw_frame(const AVFrame *frame, bool bUpdate)
{
    std::lock_guard<std::mutex> lck(m_syncLock);
#if USE_CACHE
    AVFrame *newframe = av_frame_alloc();
    av_frame_ref(newframe, frame);
    m_cache_frames.push_back(newframe);
    if (m_cache_frames.size() > 0) {
        m_start_playing = true;
    }
#else
    if (m_last_frame == nullptr) {
        m_last_frame = av_frame_alloc();
    }else{
        av_frame_unref(m_last_frame);
    }
    av_frame_ref(m_last_frame, frame);
    m_start_playing =true;
    //m_roi_pts.clear();
#endif

    videoW = frame->width;
    videoH = frame->height;
    if (bUpdate){
        //QEvent *e = new QEvent(BM_UPDATE_VIDEO);
        //QCoreApplication::postEvent(this, e);
    }
    return 0;
}

int video_opengl_widget::draw_rect(const fdrtsp::FaceRectVector& vct_rect, const std::vector<fdrtsp::FaceRecognitionInfo> &vct_label, bool bUpdate)
{
    std::lock_guard<std::mutex> lck(m_syncLock);
    if (vct_rect.size() > 0) {
        m_roi_heatbeat = 0;
        m_roi_pts.clear();
        auto w = videoW >> 1;
        auto h = videoH >> 1;

        if (w == 0 || h == 0) return 0;

        for(auto rc: vct_rect) {
            float x1, y1, x2, y2;

            x1 = (rc.x1 - w)/w;
            y1 = (h - rc.y1)/h;
            x2 = (rc.x2 - w)/w;
            y2 = (h - rc.y2)/h;

            m_roi_pts.push_back(x2); m_roi_pts.push_back(y1); m_roi_pts.push_back(0.0f);
            m_roi_pts.push_back(x1); m_roi_pts.push_back(y1); m_roi_pts.push_back(0.0f);
            m_roi_pts.push_back(x1); m_roi_pts.push_back(y2); m_roi_pts.push_back(0.0f);
            m_roi_pts.push_back(x2); m_roi_pts.push_back(y2); m_roi_pts.push_back(0.0f);
        }

        if (bUpdate) {
            //update();
        }
    }

    return 0;
}

void video_opengl_widget::init_yuv420p_shader()
{
    static const GLfloat vertices[]{
            //顶点坐标
            -1.0f,-1.0f,
            -1.0f,+1.0f,
            +1.0f,+1.0f,
            +1.0f,-1.0f,
            //纹理坐标
            0.0f,1.0f,
            0.0f,0.0f,
            1.0f,0.0f,
            1.0f,1.0f,
    };


    m_yuv420p_vbo.create();
    m_yuv420p_vbo.bind();
    m_yuv420p_vbo.allocate(vertices,sizeof(vertices));

    QOpenGLShader *vshader = new QOpenGLShader(QOpenGLShader::Vertex,this);
    const char *vsrc =
            "attribute vec4 vertexIn; \
    attribute vec2 textureIn; \
    varying vec2 textureOut;  \
    void main(void)           \
    {                         \
        gl_Position = vertexIn; \
        textureOut = textureIn; \
    }";
    vshader->compileSourceCode(vsrc);

    QOpenGLShader *fshader = new QOpenGLShader(QOpenGLShader::Fragment,this);
    const char *fsrc ="varying vec2 textureOut; \
    uniform sampler2D tex_y; \
    uniform sampler2D tex_u; \
    uniform sampler2D tex_v; \
    void main(void) \
    { \
        vec3 yuv; \
        vec3 rgb; \
        yuv.x = texture2D(tex_y, textureOut).r; \
        yuv.y = texture2D(tex_u, textureOut).r - 0.5; \
        yuv.z = texture2D(tex_v, textureOut).r - 0.5; \
        rgb = mat3( 1,       1,         1, \
                    0,       -0.39465,  2.03211, \
                    1.13983, -0.58060,  0) * yuv; \
        gl_FragColor = vec4(rgb, 1); \
    }";
    fshader->compileSourceCode(fsrc);

    m_yuv420p_program = new QOpenGLShaderProgram(this);
    m_yuv420p_program->addShader(vshader);
    m_yuv420p_program->addShader(fshader);
    m_yuv420p_program->bindAttributeLocation("vertexIn",VERTEXIN);
    m_yuv420p_program->bindAttributeLocation("textureIn",TEXTUREIN);
    m_yuv420p_program->link();

    m_yuv420p_program->enableAttributeArray(VERTEXIN);
    m_yuv420p_program->enableAttributeArray(TEXTUREIN);
    m_yuv420p_program->setAttributeBuffer(VERTEXIN,GL_FLOAT,0,2,2*sizeof(GLfloat));
    m_yuv420p_program->setAttributeBuffer(TEXTUREIN,GL_FLOAT,8*sizeof(GLfloat),2,2*sizeof(GLfloat));

    m_yuv420p_textureUniformY = m_yuv420p_program->uniformLocation("tex_y");
    m_yuv420p_textureUniformU = m_yuv420p_program->uniformLocation("tex_u");
    m_yuv420p_textureUniformV = m_yuv420p_program->uniformLocation("tex_v");
    m_yuv420p_textureY = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_yuv420p_textureU = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_yuv420p_textureV = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_yuv420p_textureY->create();
    m_yuv420p_textureU->create();
    m_yuv420p_textureV->create();
    m_yuv420p_idY = m_yuv420p_textureY->textureId();
    m_yuv420p_idU = m_yuv420p_textureU->textureId();
    m_yuv420p_idV = m_yuv420p_textureV->textureId();
}

void video_opengl_widget::init_rect_shader()
{
    GLfloat quadVertices[] = {
            0.5f, 0.5f, 0.0f,
            -0.5f, 0.5f, 0.0f,
            -0.5f,-0.5f, 0.0f,
            0.5f,-0.5f, 0.0f

    };

    QOpenGLShader *vshader = new QOpenGLShader(QOpenGLShader::Vertex,this);
    const char *vsrc =
            "attribute vec4 vertexIn;"
            "void main(void)"
            "{"
            "    gl_Position = vertexIn;"
            "}";
    vshader->compileSourceCode(vsrc);

    QOpenGLShader *fshader = new QOpenGLShader(QOpenGLShader::Fragment,this);

    const char *fsrc = "uniform vec4 vColor;"
                       "void main(void)"
                       "{ gl_FragColor = vColor;"
                       "}";

    fshader->compileSourceCode(fsrc);

    m_roi_program = new QOpenGLShaderProgram(this);
    m_roi_program->addShader(vshader);
    m_roi_program->addShader(fshader);
    m_roi_program->bindAttributeLocation("vertexIn",2);
    m_roi_program->link();

    m_roi_vbo.create();
    m_roi_vbo.bind();
    m_roi_vbo.allocate(quadVertices, sizeof(quadVertices));

    m_roi_program->enableAttributeArray(2);
    m_roi_program->setAttributeBuffer(2, GL_FLOAT, 0, 3, 3*sizeof(float));
}

void video_opengl_widget::init_nv12_shader()
{
    const char *vsrc =
            "attribute vec4 vertexIn; \
             attribute vec4 textureIn; \
             varying vec4 textureOut;  \
             void main(void)           \
             {                         \
                 gl_Position = vertexIn; \
                 textureOut = textureIn; \
             }";

    const char *fsrc =
            "varying mediump vec4 textureOut;\n"
            "uniform sampler2D textureY;\n"
            "uniform sampler2D textureUV;\n"
            "void main(void)\n"
            "{\n"
            "vec3 yuv; \n"
            "vec3 rgb; \n"
            "yuv.x = texture2D(textureY, textureOut.st).r - 0.0625; \n"
            "yuv.y = texture2D(textureUV, textureOut.st).r - 0.5; \n"
            "yuv.z = texture2D(textureUV, textureOut.st).g - 0.5; \n"
            "rgb = mat3( 1,       1,         1, \n"
            "0,       -0.39465,  2.03211, \n"
            "1.13983, -0.58060,  0) * yuv; \n"
            "gl_FragColor = vec4(rgb, 1); \n"
            "}\n";
    m_nv12_program = new QOpenGLShaderProgram(this);
    m_nv12_program->addShaderFromSourceCode(QOpenGLShader::Vertex,vsrc);
    m_nv12_program->addShaderFromSourceCode(QOpenGLShader::Fragment,fsrc);
    m_nv12_program->link();

    GLfloat points[]{
            -1.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, -1.0f,
            -1.0f, -1.0f,

            0.0f,0.0f,
            1.0f,0.0f,
            1.0f,1.0f,
            0.0f,1.0f
    };
    m_nv12_vbo.create();
    m_nv12_vbo.bind();
    m_nv12_vbo.allocate(points,sizeof(points));

    m_nv12_textureY= new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_nv12_textureUV = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_nv12_textureY->create();
    m_nv12_textureUV->create();
    m_nv12_idY = m_nv12_textureY->textureId();
    m_nv12_idUV = m_nv12_textureUV->textureId();
}

void video_opengl_widget::nv12_draw(AVFrame *frame)
{
    GLfloat points[]{
            -1.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, -1.0f,
            -1.0f, -1.0f,

            0.0f,0.0f,
            1.0f,0.0f,
            1.0f,1.0f,
            0.0f,1.0f
    };

    m_nv12_vbo.bind();
    m_nv12_vbo.allocate(points,sizeof(points));

    m_nv12_program->bind();
    m_nv12_program->enableAttributeArray("vertexIn");
    m_nv12_program->enableAttributeArray("textureIn");
    m_nv12_program->setAttributeBuffer("vertexIn",GL_FLOAT, 0, 2, 2*sizeof(GLfloat));
    m_nv12_program->setAttributeBuffer("textureIn",GL_FLOAT,2 * 4 * sizeof(GLfloat),2,2*sizeof(GLfloat));

    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D,m_nv12_idY);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RED, videoW, videoH,0,GL_RED,GL_UNSIGNED_BYTE, frame->data[0]);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, m_nv12_idUV);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RG, videoW >> 1, videoH >> 1,0,GL_RG,GL_UNSIGNED_BYTE, frame->data[1]);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    m_nv12_program->setUniformValue("textureUV",0);
    m_nv12_program->setUniformValue("textureY",1);
    glDrawArrays(GL_QUADS,0,4);
    m_nv12_program->disableAttributeArray("vertexIn");
    m_nv12_program->disableAttributeArray("textureIn");
    m_nv12_vbo.release();
    m_nv12_program->release();
}

void video_opengl_widget::initializeGL()
{
    initializeOpenGLFunctions();

    glEnable(GL_DEPTH_TEST);

    init_yuv420p_shader();
    init_nv12_shader();
    init_rect_shader();

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void video_opengl_widget::yuv420p_draw(AVFrame *frame) {
    // Use Program video
    m_yuv420p_program->bind();

//    QMatrix4x4 m;
//    m.perspective(60.0f, 4.0f/3.0f, 0.1f, 100.0f );//透视矩阵随距离的变化，图形跟着变化。屏幕平面中心就是视点（摄像头）,需要将图形移向屏幕里面一定距离。
//    m.ortho(-2,+2,-2,+2,-10,10);//近裁剪平面是一个矩形,矩形左下角点三维空间坐标是（left,bottom,-near）,右上角点是（right,top,-near）所以此处为负，表示z轴最大为10；
    //远裁剪平面也是一个矩形,左下角点空间坐标是（left,bottom,-far）,右上角点是（right,top,-far）所以此处为正，表示z轴最小为-10；
    //此时坐标中心还是在屏幕水平面中间，只是前后左右的距离已限制。
    glActiveTexture(GL_TEXTURE0);  //激活纹理单元GL_TEXTURE0,系统里面的
    glBindTexture(GL_TEXTURE_2D, m_yuv420p_idY); //绑定y分量纹理对象id到激活的纹理单元
    //使用内存中的数据创建真正的y分量纹理数据
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, videoW, videoH, 0, GL_RED, GL_UNSIGNED_BYTE, frame->data[0]);
    //https://blog.csdn.net/xipiaoyouzi/article/details/53584798 纹理参数解析
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glActiveTexture(GL_TEXTURE1); //激活纹理单元GL_TEXTURE1
    glBindTexture(GL_TEXTURE1, m_yuv420p_idU);
    //使用内存中的数据创建真正的u分量纹理数据
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, videoW >> 1, videoH >> 1, 0, GL_RED, GL_UNSIGNED_BYTE,
                 frame->data[1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glActiveTexture(GL_TEXTURE2); //激活纹理单元GL_TEXTURE2
    glBindTexture(GL_TEXTURE_2D, m_yuv420p_idV);
    //使用内存中的数据创建真正的v分量纹理数据
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, videoW >> 1, videoH >> 1, 0, GL_RED, GL_UNSIGNED_BYTE,
                 frame->data[2]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //指定y纹理要使用新值
    glUniform1i(m_yuv420p_textureUniformY, 0);
    //指定u纹理要使用新值
    glUniform1i(m_yuv420p_textureUniformU, 1);
    //指定v纹理要使用新值
    glUniform1i(m_yuv420p_textureUniformV, 2);
    //使用顶点数组方式绘制图形
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    m_yuv420p_program->release();

}

void video_opengl_widget::rect_draw() {
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    if (m_roi_pts.size() > 0) {
        glUseProgram(m_roi_program->programId());

        // DrawRectangle
        QColor color(0, 255, 0);
        glLineWidth(3);
        m_roi_program->setUniformValue("vColor", color);
        m_nv12_vbo.bind();
        m_roi_vbo.allocate(m_roi_pts.data(), m_roi_pts.size() * sizeof(float));
        m_roi_program->enableAttributeArray(2);
        m_roi_program->setAttributeBuffer(2, GL_FLOAT, 0, 3, 3 * sizeof(float));

        for(int i = 0; i< m_roi_pts.size()/3; i=i+4)
        {
            glDrawArrays(GL_LINE_LOOP, i, 4);
        }
    }

    m_roi_vbo.release();
    m_roi_program->release();
    m_roi_heatbeat++;
    if (m_roi_heatbeat > RECT_DELAY_TIME) {
        m_roi_heatbeat = 0;
        m_roi_pts.clear();
    }


    //glDisable(GL_BLEND);
}

void video_opengl_widget::paintGL() {
    std::lock_guard<std::mutex> lck(m_syncLock);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if USE_CACHE
    if (m_cache_frames.size() == 0)  {
        m_start_playing = false;
        if (m_last_frame == nullptr) {
            return;
        }
    }else{

        int64_t n = av_gettime_relative();
        int64_t delta=0;

        while(m_cache_frames.size() > 0) {
            if (m_last_frame != NULL) {
                delta = RTPUtils::rtp_time_to_ntp_us(m_cache_frames.front()->pkt_pts - m_last_frame->pkt_pts,
                                                     90000);
            }

            if (m_last_draw_time_us + delta < n || m_cache_frames.size() > 2) {
                m_last_draw_time_us = n;

                if (m_last_frame != nullptr) {
                    av_frame_unref(m_last_frame);
                    av_frame_free(&m_last_frame);
                }

                m_last_frame = m_cache_frames.front();
                m_cache_frames.pop_front();
            }else{
                break;
            }
        }
    }
#else
     if (m_last_frame == nullptr || m_is_stop_render) {
        return;
    }

#endif
    //flog(LOG_INFO, "queue frame num: %d", m_cache_frames.size());
    rect_draw();
    if (AV_PIX_FMT_NV12 == m_last_frame->format) {
        nv12_draw(m_last_frame);
    }else {
        yuv420p_draw(m_last_frame);
    }
}

void video_opengl_widget::slot_draw_rect(const fdrtsp::FaceRectVector &vct_rect, const std::vector<fdrtsp::FaceRecognitionInfo> &vct_label)
{
     draw_rect(vct_rect, vct_label);
}

bool video_opengl_widget::event(QEvent *e)
{
    if (e->type() == BM_UPDATE_VIDEO) {
        repaint();
        return true;
    }

    return QWidget::event(e);
}

void video_opengl_widget::onRefreshTimeout()
{
    if (m_start_playing) {
        repaint();
    }
}

int video_opengl_widget::clear_frame() {
    std::cout << "clear_frame()" << std::endl;
    std::lock_guard<std::mutex> lck(m_syncLock);
    av_frame_free(&m_last_frame);
    m_last_frame = nullptr;
    //m_is_stop_render = true;
    return 0;
}
