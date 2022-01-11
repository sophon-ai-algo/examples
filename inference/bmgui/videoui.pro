#-------------------------------------------------
#
# Project created by QtCreator 2017-08-16T21:22:49
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += debug

QMAKE_CXXFLAGS += -std=c++11

TARGET = videoui
TEMPLATE = app

SOURCES += main.cpp
           videoui_mainwindow.cpp


HEADERS  += videoui_mainwindow.h



FORMS    += videoui_mainwindow.ui


RESOURCES +=