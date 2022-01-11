#include "container_widget.h"

#include <iostream>
container_widget::container_widget(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);
    this->setStyleSheet("background-color:black");
}

container_widget::~container_widget()
{
}


video_widget* container_widget::addChildWnd()
{
    video_widget *pWnd = new video_widget(this);
    pWnd->setWindowFlags(Qt::SubWindow);
    connect(pWnd, SIGNAL(signal_mouse_clicked(video_widget *)), SLOT(handle_subwindow_clicked(video_widget *)));

    pWnd->show();
    mListWidgets.append(pWnd);

    UpdateWidgetLayout();

    return pWnd;
}

void container_widget::removeChildWindow(video_widget *pWnd)
{
    mListWidgets.removeOne(pWnd);
    delete pWnd;
}

void container_widget::UpdateWidgetLayout()
{
    int num = mListWidgets.count();
    if (num == 0) return;
    if (num == 1) {
        QWidget *pWidget = mListWidgets.at(0);
        pWidget->setGeometry(QRect(0, 0, width(), height()));
    }else if (num == 13) {
        int n = 4;
        const QRect &rc = geometry();
        int ww = rc.width() / n;
        int wh = rc.height() / n;

        for (int i = 0; i < num; ++i) {
            if (i < 5) {
                int row = i / n;
                int col = i % n;
                QRect rcWiget(col * ww, row * wh, ww - 1, wh - 1);
                QWidget *pWidget = mListWidgets.at(i);
                pWidget->setGeometry(rcWiget);
            }else if (i == 5) {
                QRect rcWiget(1 * ww, 1 * wh, ww*2 - 1, wh*2 - 1);
                QWidget *pWidget = mListWidgets.at(i);
                pWidget->setGeometry(rcWiget);

            }else if (i == 6) {
                QRect rcWiget(3 * ww, 1 * wh, ww - 1, wh - 1);
                QWidget *pWidget = mListWidgets.at(i);
                pWidget->setGeometry(rcWiget);
            }else if (i == 7) {
                QRect rcWiget(0 * ww, 2 * wh, ww - 1, wh - 1);
                QWidget *pWidget = mListWidgets.at(i);
                pWidget->setGeometry(rcWiget);
            }else if (i ==  8) {
                QRect rcWiget(3 * ww, 2 * wh, ww - 1, wh - 1);
                QWidget *pWidget = mListWidgets.at(i);
                pWidget->setGeometry(rcWiget);
            }else{
                int s = i - 9;
                QRect rcWiget(s * ww, 3 * wh, ww - 1, wh - 1);
                QWidget *pWidget = mListWidgets.at(i);
                pWidget->setGeometry(rcWiget);
            }
        }


    }
    else{
        int n = sqrt(num);
        if (n*n < num) n++;
        const QRect &rc = geometry();
        int ww = rc.width() / n;
        int wh = rc.height() / (num % n ==0 ? num/n: num/n + 1);

        for (int i = 0; i < num; ++i) {
            int row = i / n;
            int col = i % n;
            QRect rcWiget(col * ww, row*wh, ww-1, wh-1);

            QWidget *pWidget = mListWidgets.at(i);
            pWidget->setGeometry(rcWiget);
        }
    }

}

void container_widget::resizeEvent(QResizeEvent* size)
{
    UpdateWidgetLayout();
}

void container_widget::handle_subwindow_clicked(video_widget *pSelectedWnd)
{
    if (m_selectedWnd != pSelectedWnd) {
        pSelectedWnd->SetSate(1);
        for (auto w : mListWidgets) {
            if (w != pSelectedWnd) {
                w->SetSate(0);
            }
        }

        m_selectedWnd = pSelectedWnd;
    }
}

video_widget* container_widget::getSelectedWidget()
{
    return m_selectedWnd;
}

