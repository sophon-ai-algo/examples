#include "mainwindow.h"
#include "container_widget.h"
#include "ui_mainwindow.h"

mainwindow::mainwindow(QWidget *parent) :
    QMainWindow(parent),m_pMainWidget(nullptr),
    ui(new Ui::mainwindow)
{
    ui->setupUi(this);
    m_pMainWidget = new container_widget(this);
    setCentralWidget(m_pMainWidget);
}

mainwindow::~mainwindow()
{
    delete ui;
}

int mainwindow::createWidgets(int num) {
    for(int i = 0;i < num; i++) {
        m_mapWidgets[i] = m_pMainWidget->addChildWnd();
    }
    return 0;
}

video_widget* mainwindow::videoWidget(int id) {
    if (m_mapWidgets.find(id) != m_mapWidgets.end())
        return m_mapWidgets[id];
    else{
        return nullptr;
    }
}
