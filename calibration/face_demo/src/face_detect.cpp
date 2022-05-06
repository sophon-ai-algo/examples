#include "face_detect.hpp"

#define DYNAMIC_SIZE	(1)

void FaceDetector::initParameters(void) {
  target_size_ = 400;
  max_size_ = 800;
  nms_threshold_ = 0.3;
  base_threshold_ = 0.05;
  per_nms_topn_ = 1000;
  base_size_ = 16;
  min_size_ = 0;
  feat_stride_ = 8;

  int ratio_size = 1;

  anchor_ratios_.push_back(1.f);

  anchor_scales_.push_back(1);
  anchor_scales_.push_back(2);
  anchor_num_ = ratio_size * 2;

}
FaceDetector::FaceDetector(const std::string& proto_file, const std::string& model_file) {

#ifdef INT8_MODE
  Ufw::set_mode(Ufw::INT8);
#else
  Ufw::set_mode(Ufw::FP32);
#endif

  net_= new Net<float>(proto_file, model_file, TEST);

  initParameters();
}


void FaceDetector::calculatePyramidScale(const cv::Mat& image,
                                               cv::Mat& resizedImage) {
  cv::Mat target_img;
  image.convertTo(target_img, CV_32FC3);

  int height = target_img.rows;
  int width = target_img.cols;
  int im_size_min = std::min(height, width);
  int im_size_max = std::max(height, width);
  im_scale_ = target_size_ / im_size_min;
  if (im_scale_ * im_size_max > max_size_)
    im_scale_ = max_size_ / im_size_max;

  cv::Scalar mean_scalar = cv::Scalar(104, 117, 123);

	resizedImage = target_img - mean_scalar;

#if DYNAMIC_SIZE
  cv::resize(resizedImage,
             resizedImage,
             cv::Size(0, 0),
             im_scale_,
             im_scale_,
             cv::INTER_LINEAR);
#else
  cv::resize(resizedImage,
             resizedImage,
             cv::Size(711, 400),
             0,
             0,
             cv::INTER_LINEAR);
#endif
}

inline bool compareBBox(const FaceRect& a, const FaceRect& b) {
  return a.score > b.score;
}

void FaceDetector::generateProposals(
    const float* scores, const float* bbox_deltas, const float scale_factor,
    const int feat_factor, const int feat_w, const int feat_h, const int width,
    const int height, std::vector<FaceRect>& proposals) {
  std::vector<FaceRect> m_proposals;
  float anchor_cx = (base_size_ - 1) * 0.5;
  float anchor_cy = (base_size_ - 1) * 0.5;
  int feat_stride = feat_stride_ * feat_factor;

  for (int s = 0; s < anchor_scales_.size(); ++s) {
    float scale = anchor_scales_[s] * scale_factor;
    for (int h = 0; h < feat_h; ++h) {
      for (int w = 0; w < feat_w; ++w) {
        int delta_index = h * feat_w + w;
        FaceRect facerect;
        facerect.score = scores[s * feat_w * feat_h + delta_index];
        if (facerect.score <= base_threshold_) continue;
        float anchor_size = scale * base_size_;
        float bbox_x1 = anchor_cx - (anchor_size - 1) * 0.5 + w * feat_stride;
        float bbox_y1 = anchor_cy - (anchor_size - 1) * 0.5 + h * feat_stride;
        float bbox_x2 = anchor_cx + (anchor_size - 1) * 0.5 + w * feat_stride;
        float bbox_y2 = anchor_cy + (anchor_size - 1) * 0.5 + h * feat_stride;

        float bbox_w = bbox_x2 - bbox_x1 + 1;
        float bbox_h = bbox_y2 - bbox_y1 + 1;
        float bbox_cx = bbox_x1 + 0.5 * bbox_w;
        float bbox_cy = bbox_y1 + 0.5 * bbox_h;
        float dx = bbox_deltas[(s * 4 + 0) * feat_h * feat_w + delta_index];
        float dy = bbox_deltas[(s * 4 + 1) * feat_h * feat_w + delta_index];
        float dw = bbox_deltas[(s * 4 + 2) * feat_h * feat_w + delta_index];
        float dh = bbox_deltas[(s * 4 + 3) * feat_h * feat_w + delta_index];
        float pred_cx = dx * bbox_w + bbox_cx;
        float pred_cy = dy * bbox_h + bbox_cy;
        float pred_w = std::exp(dw) * bbox_w;
        float pred_h = std::exp(dh) * bbox_h;
        facerect.x1 = std::max(std::min(static_cast<double>(width - 1),
                                        (pred_cx - 0.5 * pred_w) / im_scale_),
                               0.0);
        facerect.y1 = std::max(std::min(static_cast<double>(height - 1),
                                        (pred_cy - 0.5 * pred_h) / im_scale_),
                               0.0);
        facerect.x2 = std::max(std::min(static_cast<double>(width - 1),
                                        (pred_cx + 0.5 * pred_w) / im_scale_),
                               0.0);
        facerect.y2 = std::max(std::min(static_cast<double>(height - 1),
                                        (pred_cy + 0.5 * pred_h) / im_scale_),
                               0.0);
        if ((facerect.x2 - facerect.x1 + 1 < min_size_) ||
            (facerect.y2 - facerect.y1 + 1 < min_size_))
          continue;
        m_proposals.push_back(facerect);
      }
    }
  }
  std::sort(m_proposals.begin(), m_proposals.end(), compareBBox);

  int keep = m_proposals.size();
  if (per_nms_topn_ < keep) keep = per_nms_topn_;

  if (keep > 0) {
    proposals.insert(proposals.end(), m_proposals.begin(),
                     m_proposals.begin() + keep);
  }
}

void FaceDetector::nms(const std::vector<FaceRect>& proposals,
                             std::vector<FaceRect>& nmsProposals) {
  if (proposals.empty()) {
    nmsProposals.clear();
    return;
  }
  std::vector<FaceRect> bboxes = proposals;
  std::sort(bboxes.begin(), bboxes.end(), compareBBox);

  int select_idx = 0;
  int num_bbox = bboxes.size();
  std::vector<int> mask_merged(num_bbox, 0);
  bool all_merged = false;
  while (!all_merged) {
    while (select_idx < num_bbox && 1 == mask_merged[select_idx]) ++select_idx;

    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }
    nmsProposals.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;
    FaceRect select_bbox = bboxes[select_idx];
    float area1 = (select_bbox.x2 - select_bbox.x1 + 1) *
                  (select_bbox.y2 - select_bbox.y1 + 1);
    ++select_idx;
    for (int i = select_idx; i < num_bbox; ++i) {
      if (mask_merged[i] == 1) continue;
      FaceRect& bbox_i = bboxes[i];
      float x = std::max(select_bbox.x1, bbox_i.x1);
      float y = std::max(select_bbox.y1, bbox_i.y1);
      float w = std::min(select_bbox.x2, bbox_i.x2) - x + 1;
      float h = std::min(select_bbox.y2, bbox_i.y2) - y + 1;
      if (w <= 0 || h <= 0) continue;
      float area2 = (bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1);
      float area_intersect = w * h;
      // Union method
      if (area_intersect / (area1 + area2 - area_intersect) > nms_threshold_)
        mask_merged[i] = 1;
    }
  }
}

void FaceDetector::detect(const cv::Mat& image, const float threshold,
                                std::vector<FaceRect>& faceRects) {
  cv::Mat resized;
  calculatePyramidScale(image, resized);

  int net_h = resized.rows;
  int net_w = resized.cols;
  int net_c = resized.channels();
  int net_b = 1;
  std::cout << "net input shape <" << net_b << "," << net_c << "," << net_h
            << "," << net_w << ">" << std::endl;
  
  //reshape input blob
  Blob<float> *input_blob = (net_-> blob_by_name("data")).get();
  input_blob->Reshape(net_b, net_c, net_h, net_w);
  
  //fill data
  input_blob->universe_fill_data(resized);

  std::cout << "Running for " <<  Ufw::mode() << " mode for squeezenet"<<std::endl;
  net_->Forward();

  std::vector<FaceRect> proposals;
  int width = image.cols;
  int height = image.rows;

  Blob<float>* m3_cls_tensor =
      net_->blob_by_name("m3@ssh_cls_prob_reshape_output").get();
  
  Blob<float>* m3_bbox_tensor =
      net_->blob_by_name("m3@ssh_bbox_pred_output").get();


  int m3_w = m3_cls_tensor->width();
  int m3_h = m3_cls_tensor->height();
  const float* m3_scores = m3_cls_tensor->universe_get_data();
  m3_scores += anchor_num_ * m3_w * m3_h;
  
  const float* m3_bbox_deltas = m3_bbox_tensor->universe_get_data();
  generateProposals(m3_scores, m3_bbox_deltas, 16.0, 4, m3_w,
                    m3_h, width, height, proposals);

  Blob<float>* m2_cls_tensor =
      net_->blob_by_name("m2@ssh_cls_prob_reshape_output").get();
  Blob<float>* m2_bbox_tensor =
      net_->blob_by_name("m2@ssh_bbox_pred_output").get();


  int m2_w = m2_cls_tensor->width();
  int m2_h = m2_cls_tensor->height();
  const float* m2_scores = m2_cls_tensor->universe_get_data();
  m2_scores += anchor_num_ * m2_w * m2_h;
  const float* m2_bbox_deltas = m2_bbox_tensor->universe_get_data();
  generateProposals(m2_scores, m2_bbox_deltas, 4.0, 2, m2_w,
                    m2_h, width, height, proposals);

  Blob<float>* m1_cls_tensor =
      net_->blob_by_name("m1@ssh_cls_prob_reshape_output").get();
  Blob<float>* m1_bbox_tensor =
      net_->blob_by_name("m1@ssh_bbox_pred_output").get();

  int m1_w = m1_cls_tensor->width();
  int m1_h = m1_cls_tensor->height();
  const float* m1_scores = m1_cls_tensor->universe_get_data();
  m1_scores += anchor_num_ * m1_w * m1_h;
  const float* m1_bbox_deltas = m1_bbox_tensor->universe_get_data();
  generateProposals(m1_scores, m1_bbox_deltas, 1.0, 1, m1_w,
                    m1_h, width, height, proposals);


  std::vector<FaceRect> nmsProposals;
  nms(proposals, nmsProposals);

  faceRects.clear();
  for (int i = 0; i < nmsProposals.size(); ++i) {
    FaceRect rect = nmsProposals[i];
    if (rect.score >= threshold) faceRects.push_back(rect);
  }
  std::cout << "final predict " << faceRects.size() << " bboxes" << std::endl;
}
