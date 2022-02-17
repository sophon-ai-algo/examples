#ifndef VIDEO_WIDGET_H
#define VIDEO_WIDGET_H

#include <QWidget>
#include <memory>
#include "video_pixmap_widget.h"

#ifdef USE_OPENGL_RENDER
#include "video_opengl_widget.h"
#endif

namespace Ui {
class video_widget;
}


enum {
    ASPECT_FIT  =0,
    FULL_FIT    =1
};

class video_widget : public QWidget
{
    Q_OBJECT

    QTimer *m_timer;
    QRect m_rcContainer;
    bool  m_dragEnabled;
    int   m_state{ 0 };
    int   m_fit_mode;
public:
    explicit video_widget(QWidget *parent = 0, int fit_mode=FULL_FIT);
    ~video_widget();

    void enableDrag(bool enabled) {
        m_dragEnabled = enabled;
    }

    IVideoDrawer* GetVideoHwnd();

    void SetTitle(QString strTitle);
    void VideoFitByRatio(const QRect& rcContainer, int w, int h);

    void SetSate(int state);

signals:
    void signal_mouse_clicked(video_widget *pSelf);

protected:
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    //virtual void showEvent(QShowEvent *event);
    virtual void paintEvent(QPaintEvent *event);
    virtual void resizeEvent(QResizeEvent *event);
    virtual void mouseDoubleClickEvent(QMouseEvent *) override;
private slots:
    void onTimeout();
    void drawVideoBoarder(QPainter &p);

private:
    bool        mMoveing;
    QPoint      mMovePosition;
    bool        mTitleEnabled{false};
#ifdef USE_OPENGL_RENDER
    video_opengl_widget *m_video;
#else
    video_pixmap_widget *m_video;
#endif
    bool        mIsFullScreen{false};
    QRect       m_originWindowRect;

private:
    Ui::video_widget *ui;
};

#endif // VIDEO_WIDGET_H
