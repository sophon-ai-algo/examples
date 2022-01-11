#pragma once

#include <QWidget>
#include <QtCore>
#include "ui_container_widget.h"
#include "video_widget.h"

class container_widget : public QWidget
{
    Q_OBJECT

public:
    container_widget(QWidget *parent = Q_NULLPTR);
    ~container_widget();

    video_widget* addChildWnd();
    void removeChildWindow(video_widget *pWnd);
    void UpdateWidgetLayout();

    video_widget* getSelectedWidget();
protected:
    void resizeEvent(QResizeEvent* size);
protected slots:
    void handle_subwindow_clicked(video_widget *pSelectedWnd);
private:
    QList<video_widget*> mListWidgets;

private:
    video_widget *m_selectedWnd{ 0 };
    Ui::container_widget ui;
};
