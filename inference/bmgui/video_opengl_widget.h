//
// Created by hsyuan on 2019-03-20.
//

#ifndef FACEDEMOEXAMPLE_VIDEO_RENDER_OPENGL_H
#define FACEDEMOEXAMPLE_VIDEO_RENDER_OPENGL_H

#include <QtWidgets>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QTimer>
#include <mutex>
#include "video_pixmap_widget.h"

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

class video_opengl_widget : public QOpenGLWidget, protected QOpenGLFunctions, public IVideoDrawer
{
Q_OBJECT
public:
    video_opengl_widget(QWidget *parent =0);
    ~video_opengl_widget();

    //Implement IVideoDrawer
    int frame_width() override;
    int frame_height() override;
    int draw_frame(const AVFrame *frame, bool bUpdate=true) override;
    //int draw_rect(const bm::NetOutputObjectArray& vct_rect, bool bUpdate=true) override;
    int clear_frame();

public slots:
    void slot_draw_frame(const AVFrame *frame); //显示一帧Yuv图像
    void slot_draw_rect(const bm::NetOutputObjects& vct_rect);

protected:
    bool event(QEvent *e) override;
    void init_yuv420p_shader();
    void yuv420p_draw(AVFrame *frame);

    void init_nv12_shader();
    void nv12_draw(AVFrame *frame);

    void init_rect_shader();
    void rect_draw();

    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;

protected slots:
    void onRefreshTimeout();

private:
    QOpenGLShaderProgram *m_yuv420p_program;
    QOpenGLShaderProgram *m_nv12_program;
    QOpenGLShaderProgram *m_roi_program;

    //420p
    QOpenGLBuffer m_yuv420p_vbo;
    GLuint m_yuv420p_textureUniformY, m_yuv420p_textureUniformU,m_yuv420p_textureUniformV;
    QOpenGLTexture *m_yuv420p_textureY = nullptr,*m_yuv420p_textureU = nullptr,*m_yuv420p_textureV = nullptr;

    GLuint m_yuv420p_idY,m_yuv420p_idU,m_yuv420p_idV;

    //nv12
    QOpenGLBuffer m_nv12_vbo;
    GLuint m_nv12_textureUniformY, m_nv12_textureUniformUV;
    QOpenGLTexture *m_nv12_textureY=nullptr, *m_nv12_textureUV= nullptr;
    GLuint m_nv12_idY,m_nv12_idUV;

    uint videoW,videoH;

    // Test
    GLuint m_posAttr;
    QOpenGLBuffer m_roi_vbo;
    QVector<GLfloat> m_roi_pts;

    QTimer *m_refreshTimer;
    std::mutex m_syncLock;
    bool m_start_playing;
    AVFrame *m_last_frame;
    int64_t m_last_draw_time_us;
    int m_roi_heatbeat{0};

    bool m_is_stop_render{false};

};


#endif //FACEDEMOEXAMPLE_VIDEO_RENDER_OPENGL_H
