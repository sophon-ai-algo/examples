#include "face_util.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
inline float calc_sqrt(const std::vector<float>& feature) {
  float a = 0;
  for (int i = 0; i < feature.size(); ++i) {
    a += feature[i] * feature[i];
  }
  return sqrt(a);
}

float calc_cosine(const std::vector<float>& feature1,
                  const std::vector<float>& feature2) {
  float a = 0;
  for (int i = 0; i < feature1.size(); ++i) {
    a += feature1[i] * feature2[i];
  }
  float b = calc_sqrt(feature1) * calc_sqrt(feature2);
  return a / b;
}

inline float calc_sqrt(const std::vector<char>& feature) {
  float a = 0;
  for (int i = 0; i < feature.size(); ++i) {
    a += ((float)feature[i]) * ((float)feature[i]);
  }
  return sqrt(a);
}

float calc_cosine(const std::vector<char>& feature1,
                  const std::vector<char>& feature2) {
  float a = 0;
  for (int i = 0; i < feature1.size(); ++i) {
    a += ((float)feature1[i]) * ((float)feature2[i]);
  }
  float b = calc_sqrt(feature1) * calc_sqrt(feature2);
  return a / b;
}

float calc_cosine(const std::vector<char>& feature1,
                  const std::vector<float>& feature2) {
  float a = 0;
  for (int i = 0; i < feature1.size(); ++i) {
    a += ((float)feature1[i]) * feature2[i];
  }
  float b = calc_sqrt(feature1) * calc_sqrt(feature2);
  return a / b;
}

bool compareArea(const FaceRect& a, const FaceRect& b) {
  return (a.x2 - a.x1) * (a.y2 - a.y1) > (b.x2 - b.x1) * (b.y2 - b.y1);
}

bool compareScore(const FaceRect& a, const FaceRect& b) {
  return a.score > b.score;
}

cv::Mat tformfwd(const cv::Mat& trans, const cv::Mat& uv) {
  cv::Mat uv_h = cv::Mat::ones(uv.rows, 3, CV_64FC1);
  uv.copyTo(uv_h(cv::Rect(0, 0, 2, uv.rows)));
  cv::Mat xv_h = uv_h * trans;
  return xv_h(cv::Rect(0, 0, 2, uv.rows));
}

cv::Mat find_none_flectives_similarity(const cv::Mat& uv, const cv::Mat& xy) {
  cv::Mat A = cv::Mat::zeros(2 * xy.rows, 4, CV_64FC1);
  cv::Mat b = cv::Mat::zeros(2 * xy.rows, 1, CV_64FC1);
  cv::Mat x = cv::Mat::zeros(4, 1, CV_64FC1);

  xy(cv::Rect(0, 0, 1, xy.rows)).copyTo(A(cv::Rect(0, 0, 1, xy.rows)));  // x
  xy(cv::Rect(1, 0, 1, xy.rows)).copyTo(A(cv::Rect(1, 0, 1, xy.rows)));  // y
  A(cv::Rect(2, 0, 1, xy.rows)).setTo(1.);

  xy(cv::Rect(1, 0, 1, xy.rows))
      .copyTo(A(cv::Rect(0, xy.rows, 1, xy.rows)));  // y
  xy(cv::Rect(0, 0, 1, xy.rows))
      .copyTo(A(cv::Rect(1, xy.rows, 1, xy.rows)));  //-x
  A(cv::Rect(1, xy.rows, 1, xy.rows)) *= -1;
  A(cv::Rect(3, xy.rows, 1, xy.rows)).setTo(1.);

  uv(cv::Rect(0, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, 0, 1, uv.rows)));
  uv(cv::Rect(1, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, uv.rows, 1, uv.rows)));

  cv::solve(A, b, x, cv::DECOMP_SVD);
  cv::Mat trans_inv = (cv::Mat_<double>(3, 3) << x.at<double>(0),
                       -x.at<double>(1), 0, x.at<double>(1), x.at<double>(0), 0,
                       x.at<double>(2), x.at<double>(3), 1);
  cv::Mat trans = trans_inv.inv(cv::DECOMP_SVD);
  trans.at<double>(0, 2) = 0;
  trans.at<double>(1, 2) = 0;
  trans.at<double>(2, 2) = 1;
  return trans;
}

cv::Mat find_similarity(const cv::Mat& uv, const cv::Mat& xy) {
  cv::Mat trans1 = find_none_flectives_similarity(uv, xy);
  cv::Mat xy_reflect = xy;
  xy_reflect(cv::Rect(0, 0, 1, xy.rows)) *= -1;
  cv::Mat trans2r = find_none_flectives_similarity(uv, xy_reflect);
  cv::Mat reflect = (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

  cv::Mat trans2 = trans2r * reflect;
  cv::Mat xy1 = tformfwd(trans1, uv);
  double norm1 = cv::norm(xy1 - xy);

  cv::Mat xy2 = tformfwd(trans2, uv);
  double norm2 = cv::norm(xy2 - xy);

  cv::Mat trans;
  if (norm1 < norm2) {
    trans = trans1;
  } else {
    trans = trans2;
  }
  return trans;
}

cv::Mat get_similarity_transform(const std::vector<cv::Point2f>& src_points,
                                 const std::vector<cv::Point2f>& dst_points,
                                 bool reflective = true) {
  cv::Mat trans;
  cv::Mat src((int)src_points.size(), 2, CV_32FC1, (void*)(&src_points[0].x));
  src.convertTo(src, CV_64FC1);

  cv::Mat dst((int)dst_points.size(), 2, CV_32FC1, (void*)(&dst_points[0].x));
  dst.convertTo(dst, CV_64FC1);

  if (reflective) {
    trans = find_similarity(src, dst);
  } else {
    trans = find_none_flectives_similarity(src, dst);
  }
  cv::Mat trans_cv2 = trans(cv::Rect(0, 0, 2, trans.rows)).t();
  return trans_cv2;
}


void remap_nearest(const Mat& _src, Mat& _dst, Mat& mapx)
{
    CV_Assert(_src.depth() == CV_8U && _dst.type() == _src.type());
    CV_Assert(mapx.type() == CV_16SC2);

    Size ssize = _src.size(), dsize = _dst.size();
    CV_Assert(ssize.area() > 0 && dsize.area() > 0);
    int cn = _src.channels();
    std::cout << "src size " << _src.total() << " channels " << _src.channels() << " depth " << _src.depth() << " rows " << _src.size().height << " cols " << _src.size().width << std::endl; 
    std::cout << "dst size " << _dst.total() << " channels " << _dst.channels() << " depth " << _dst.depth() << " rows " << _dst.size().height << " cols " << _dst.size().width << std::endl; 

    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const short* yM = mapx.ptr<short>(dy);
        unsigned char* yD = _dst.ptr<unsigned char>(dy);

        for (int dx = 0; dx < dsize.width; ++dx)
        {
            unsigned char* xyD = yD + cn * dx;
            int sx = yM[dx * 2], sy = yM[dx * 2 + 1];

            if (sx >= 0 && sx < ssize.width && sy >= 0 && sy < ssize.height)
            {
                const unsigned char *xyS = _src.ptr<unsigned char>(sy) + sx * cn;

                for (int r = 0; r < cn; ++r)
                    xyD[r] = xyS[r];
            }
            else
            {
                for (int r = 0; r < cn; ++r)
                    xyD[r] = saturate_cast<unsigned char>(0);
            }
        }
    }
}


void map_write(const Mat& map, std::string fn)
{
  CV_Assert(map.depth() == CV_16S);
  Size size = map.size();
  std::ofstream fout(fn, ios::out);
  if(fout.is_open())
  {
    for (int dy = 0; dy < size.height; dy++) {
        const short* yM = map.ptr<short>(dy);
        for (int dx = 0; dx < size.width; dx++, yM += 2) {
          fout << (*yM) << " ";
        }
        fout << std::endl;
    }
    fout << std::endl;

    for (int dy = 0; dy < size.height; dy++) {
        const short* yM = map.ptr<short>(dy) + 1;
        for (int dx = 0; dx < size.width; dx++, yM += 2) {
          fout << (*yM) << " ";
        }
        fout << std::endl;
    }
    fout << std::endl;

  }
  fout.close();
}


static void bmcv_warp_affine_ref(const Mat& _src, Mat& _dst, const Mat& M, Size dsize)
{
    CV_Assert(_src.size().area() > 0);
    _dst.create(dsize, _src.type());
    CV_Assert(dsize.area() > 0);
    
    const double* data_M = M.ptr<double>(0);
    std::cout << "M ";
    for (int i = 0; i < M.total(); i++) {
        std::cout << data_M[i] << " ";
    }
    std::cout << std::endl;

    Mat tM;
    M.convertTo(tM, CV_32F);
    invertAffineTransform(tM.clone(), tM);

    const float* data_tM = tM.ptr<float>(0);
    std::cout << "tM ";
    for (int i = 0; i < tM.total(); i++) {
        std::cout << data_tM[i] << " ";
    }
    std::cout << std::endl;

    Mat mapx;
    mapx.create(dsize, CV_16SC2);
      
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        short* yM = mapx.ptr<short>(dy);
        for (int dx = 0; dx < dsize.width; ++dx, yM += 2)
        {
            float sx = data_tM[0] * dx + data_tM[1] * dy + data_tM[2];
            float sy = data_tM[3] * dx + data_tM[4] * dy + data_tM[5];
            yM[0] = saturate_cast<short>(sx);
            yM[1] = saturate_cast<short>(sy);

        }
    }

    remap_nearest(_src, _dst, mapx);
}


cv::Mat align_face(const cv::Mat& src, const FaceRect rect,
                   const FacePts facePt, int width, int height) {
  const int ReferenceWidth = 96;
  const int ReferenceHeight = 112;
  std::vector<cv::Point2f> detect_points;
  for (int j = 0; j < 5; ++j) {
    cv::Point2f e;
    e.x = facePt.x[j] + rect.x1;
    e.y = facePt.y[j] + rect.y1;
    detect_points.push_back(e);
  }
  std::vector<cv::Point2f> reference_points;
  reference_points.push_back(cv::Point2f(30.29459953, 51.69630051));
  reference_points.push_back(cv::Point2f(65.53179932, 51.50139999));
  reference_points.push_back(cv::Point2f(48.02519989, 71.73660278));
  reference_points.push_back(cv::Point2f(33.54930115, 92.36550140));
  reference_points.push_back(cv::Point2f(62.72990036, 92.20410156));
  for (int j = 0; j < 5; ++j) {
    reference_points[j].x += (width - ReferenceWidth) / 2.0f;
    reference_points[j].y += (height - ReferenceHeight) / 2.0f;
  }
  cv::Mat tfm = get_similarity_transform(detect_points, reference_points);
  cv::Mat aligned_face;

  bmcv_warp_affine_ref(src, aligned_face, tfm, cv::Size(width, height));

  return aligned_face;
}
