//
// Created by hsyuan on 2019-08-01.
//

#ifndef PROJECT_QTGUI_UTILS_H
#define PROJECT_QTGUI_UTILS_H

#include <QPushButton>
#include <QImage>
#include <QLabel>

class QtGuiUtils {
public:
    static void Button_SetIcon(QPushButton *btn, const QString &path, QString toolTip, bool usePngSize) {
            QPixmap *pixmap = new QPixmap(path);
            btn->setText("");
            btn->setBackgroundRole(QPalette::Window);
            std::cout << "Pixmap:size(w=" << pixmap->width() << ", h=" << pixmap->height() << std::endl;
            std::cout << "button:size(w=" << btn->width() << ", h=" << btn->height() << std::endl;
            if (usePngSize) {
                    btn->setFixedSize(pixmap->width(), pixmap->height());
                    btn->setIcon(*pixmap);
                    btn->setIconSize(QSize(pixmap->width(), pixmap->height()));

            } else {
                    pixmap->scaled(btn->size(), Qt::KeepAspectRatio);
                    btn->setIcon(*pixmap);
                    btn->setIconSize(QSize(btn->width(), btn->height()));
            }
            btn->setToolTip(toolTip);
            btn->setStyleSheet("QPushButton{background:transparent;border:none;} QToolTip{background-color:white}");
            btn->setCursor(Qt::PointingHandCursor);
            btn->setFocusPolicy(Qt::NoFocus);
            btn->setFlat(false);

    }

    static void Label_SetIcon(QLabel *w, const QString &path, QString toolTip, bool usePngSize) {
            QPixmap *pixmap = new QPixmap(path);
            w->setText("");
            w->setBackgroundRole(QPalette::Window);
            std::cout << "Pixmap:size(w=" << pixmap->width() << ", h=" << pixmap->height() << std::endl;
            std::cout << "button:size(w=" << w->width() << ", h=" << w->height() << std::endl;
            if (usePngSize) {
                    w->setFixedSize(pixmap->width(), pixmap->height());
                    w->setPixmap(*pixmap);

            } else {
                    pixmap->scaled(w->size(), Qt::KeepAspectRatio);
                    w->setScaledContents(true);
                    w->setPixmap(*pixmap);

            }
            w->setToolTip(toolTip);
            w->setStyleSheet("QPushButton{background:transparent;border:none;} QToolTip{background-color:white}");
            w->setCursor(Qt::PointingHandCursor);
            w->setFocusPolicy(Qt::NoFocus);

    }

    static QImage cvmat_to_qimage(const cv::Mat &mat ) {
        switch ( mat.type() )
        {
            // 8-bit, 4 channel
            case CV_8UC4:
            {
                QImage image( mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB32 );
                return image;
            }

                // 8-bit, 3 channel
            case CV_8UC3:
            {
                QImage image( mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888 );
                return image.rgbSwapped();
            }

                // 8-bit, 1 channel
            case CV_8UC1:
            {
                static QVector<QRgb>  sColorTable;
                // only create our color table once
                if ( sColorTable.isEmpty() )
                {
                    for ( int i = 0; i < 256; ++i )
                        sColorTable.push_back( qRgb( i, i, i ) );
                }
                QImage image( mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8 );
                image.setColorTable( sColorTable );
                return image;
            }

            default:
                qDebug("Image format is not supported: depth=%d and %d channels\n", mat.depth(), mat.channels());
                break;
        }
        return QImage();
    }

};


#endif //PROJECT_QTGUI_UTILS_H
