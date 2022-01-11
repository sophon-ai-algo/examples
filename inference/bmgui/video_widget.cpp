#include "video_widget.h"
#include "ui_video_widget.h"
#include <QMouseEvent>
#include <QTimer>

#include "video_render.h"
#include <QtCore>
#include <QtGui>
#include <qstyle.h>
#include <qstyleoption.h>

#define BGCLOLOR 0x464646

video_widget::video_widget(QWidget *parent, int fit_mode) :
    QWidget(parent),
    m_timer(nullptr),
    m_dragEnabled(false),
    m_fit_mode(fit_mode),
    ui(new Ui::video_widget)
{
    ui->setupUi(this);
    setWindowFlags(Qt::SubWindow);
    setStyleSheet("background-color:black");

#ifdef USE_OPENGL_RENDER
    m_video = new video_opengl_widget(this);
#else
    m_video = new video_pixmap_widget(this);
#endif

    mMoveing=false;
    //Qt::FramelessWindowHint 无边框
    //Qt::WindowStaysOnTopHint 窗口在最顶端，不会拖到任务栏下面
    setWindowFlags(Qt::FramelessWindowHint | Qt::WindowMinimizeButtonHint |Qt::WindowStaysOnTopHint);

    m_video->setStyleSheet("background-color:#464646");
    m_video->show();

    if (mTitleEnabled) {
        ui->title->setStyleSheet("background-color:#4a708b");
        ui->title->setFixedHeight(20);
    }else{
        ui->title->setHidden(true);
    }

    m_timer = new QTimer(this);
    connect(m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    m_timer->start(1000);

}

video_widget::~video_widget()
{
    delete ui;
}


//重写鼠标按下事件
void video_widget::mousePressEvent(QMouseEvent *event)
{
    mMoveing = true;
    //记录下鼠标相对于窗口的位置
    //event->globalPos()鼠标按下时，鼠标相对于整个屏幕位置
    //pos() this->pos()鼠标按下时，窗口相对于整个屏幕位置
    mMovePosition = event->globalPos() - pos();
    emit signal_mouse_clicked(this);
    return QWidget::mousePressEvent(event);
}

void video_widget::mouseMoveEvent(QMouseEvent *event)
{
    if (m_dragEnabled) {
        //(event->buttons() && Qt::LeftButton)按下是左键
        //鼠标移动事件需要移动窗口，窗口移动到哪里呢？就是要获取鼠标移动中，窗口在整个屏幕的坐标，然后move到这个坐标，怎么获取坐标？
        //通过事件event->globalPos()知道鼠标坐标，鼠标坐标减去鼠标相对于窗口位置，就是窗口在整个屏幕的坐标
        if (mMoveing && (event->buttons() & Qt::LeftButton)
            && (event->globalPos() - mMovePosition).manhattanLength() > QApplication::startDragDistance())
        {
            move(event->globalPos() - mMovePosition);
            mMovePosition = event->globalPos() - pos();
        }
    }
    return QWidget::mouseMoveEvent(event);
}

void video_widget::mouseReleaseEvent(QMouseEvent *event)
{
    mMoveing = false;
}


IVideoDrawer* video_widget::GetVideoHwnd() {
    return m_video;
}

void video_widget::SetTitle(QString strTitle) {
    ui->title->setText(strTitle);
}

void video_widget::onTimeout()
{
    int w = m_video->frame_width();
    int h = m_video->frame_height();

    if (w == 0 || 0 == h) {
        w = 1280; h = 720;
    }

    int H = 20;
    if (mTitleEnabled) {
        QRect rcTitle;
        rcTitle.setRect(0, 0, geometry().width(), H);
        rcTitle.adjust(1, 1, -1, -1);
        ui->title->setGeometry(rcTitle);
        m_rcContainer = QRect(0, H, geometry().width(), geometry().height() - H);
        m_rcContainer.adjust(2, 2, -2, -2);
    }else{
        H = 0;
        m_rcContainer = QRect(0, H, geometry().width(), geometry().height() - H);
        m_rcContainer.adjust(0, 0, 0, 0);
    }

    VideoFitByRatio(m_rcContainer, w, h);
}

void video_widget::VideoFitByRatio(const QRect& rcContainer, int w, int h)
{
    int screenW = rcContainer.width();
    int screenH = rcContainer.height();

    int finalW, finalH;
    QRect rcVideo;
    int x=0, y=0;
    if(m_fit_mode == ASPECT_FIT) {
        finalH = screenW * h / w;
        if (finalH > screenH) {
            //base on H
            finalH = screenH;
            finalW = screenH * w / h;
        } else {
            finalW = screenW;
        }

        x = rcContainer.left() + (screenW - finalW) / 2;
        y = rcContainer.top() + (screenH - finalH) / 2;
    }else{
        finalW = screenW;
        finalH = screenH;
        x = 0;y = 0;
    }

    rcVideo.setRect(x, y, finalW - 1, finalH - 1);
    m_video->setGeometry(rcVideo);
    m_video->show();
}

void video_widget::paintEvent(QPaintEvent *event)
{
    QPainter p(this);
    drawVideoBoarder(p);
}

void video_widget::SetSate(int state) {
    if (m_state != state) {
        m_state = state;
        update();
    }
}

void video_widget::drawVideoBoarder(QPainter &p)
{
    int H = 0;
    if (mTitleEnabled){
        H = 20;
    }

    //draw edage
    if (m_state > 0) {
        QPen mypen;
        mypen.setColor(Qt::red);
        p.setPen(mypen);
        auto rc = QRect(0, H, geometry().width(), geometry().height() - H);
        rc.adjust(1, 1, -1, -1);
        p.drawRect(rc);
    }
    else {
        QPen mypen;
        mypen.setColor(BGCLOLOR);
        p.setPen(mypen);
        auto rc = QRect(0, H, geometry().width(), geometry().height() - H);
        rc.adjust(1, 1, -1, -1);
        p.drawRect(rc);
    }
}

void video_widget::resizeEvent(QResizeEvent *event)
{
    onTimeout();

}

void video_widget::mouseDoubleClickEvent(QMouseEvent *) {
    if (!mIsFullScreen) {
        m_originWindowRect = geometry();
        setWindowFlags(Qt::Window);
        showFullScreen();
        mIsFullScreen = true;
    }else{
        setWindowFlags(Qt::SubWindow);
        showNormal();
        setGeometry(m_originWindowRect);
        mIsFullScreen = false;
    }
}
