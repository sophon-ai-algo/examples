#include "boost/make_shared.hpp"
#include "face.hpp"
#include <ufw/ufw.hpp>
#include <ufw/layers/memory_data_layer.hpp>
#include <stdio.h>
#include <sys/stat.h>
using namespace ufw;
typedef Blob<float> __BLOB;
typedef Net<float> __NET;

class MTCNN 
{
public:
  MTCNN(const std::string& model_list);
  bool MultiDetect(const cv::Mat image, std::vector<FaceInfo> &faceInfo, int minSize, double *threshold, double factor,bool bextractFeature = false);
  void ExtractFeaturesInit(int max_iterations);
private:
    bool CvMatToDatumSignalChannel(const cv::Mat &cv_mat, Datum *datum);
    void WrapInputLayer(std::vector<cv::Mat> *input_channels, __BLOB *input_blob, const int height, const int width);
    std::vector<FaceInfo> MultiNMS(std::vector<FaceInfo> &bboxes, float thresh, char methodType, int imgsize);
    void GenerateBoundingBox(__BLOB *confidence, __BLOB *reg, float scale, float thresh, int image_width, int image_height, int imgid);
    void Bbox2Square(std::vector<FaceInfo> &bboxes);
    std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);
    void Padding(const std::vector<cv::Mat> &sample_singles);
    void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, const std::vector<cv::Mat> &sample_singles, __NET *net, double thresh, char netName);
    
    int saveImagTofile(const cv::Mat &imageData,string &savedName);
    int saveBlobTofile(__BLOB *blobPtr,string &savedName);
    void logOutChanels(std::vector<cv::Mat> &input_channels,string &fileName);

private:
    __NET * PNet_;
    __NET * RNet_;
    __NET * ONet_;
    std::vector<FaceInfo> condidate_rects_;
    std::vector<FaceInfo> total_boxes_;
    std::vector<FaceInfo> regressed_rects_;
    std::vector<FaceInfo> regressed_pading_;
};

bool CompareBBox(const FaceInfo& a, const FaceInfo& b) {
    return a.bbox.score > b.bbox.score;
}

std::vector<FaceInfo> MTCNN::MultiNMS(std::vector<FaceInfo>& BBoxes,
                                      float thresh, char methodType, int imgsize) {
    std::vector<FaceInfo> bboxes_nms;
    for(int imgid = 0; imgid < imgsize; imgid++) {
        std::vector<FaceInfo> bboxes;
        if(imgsize == 1){
            bboxes = BBoxes;
        }else {
            for(uint32_t i = 0; i < BBoxes.size(); i++) {
                if(BBoxes[i].imgid == imgid)
                    bboxes.push_back(BBoxes[i]);
            }
        }
        std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

        int32_t select_idx = 0;
        int32_t num_bbox = static_cast<int32_t>(bboxes.size());
        std::vector<int32_t> mask_merged(num_bbox, 0);
        bool all_merged = false;

        while (!all_merged) {
            while (select_idx < num_bbox && mask_merged[select_idx] == 1)
                select_idx++;
            if (select_idx == num_bbox) {
                all_merged = true;
                continue;
            }

            bboxes_nms.push_back(bboxes[select_idx]);
            mask_merged[select_idx] = 1;

            FaceRect select_bbox = bboxes[select_idx].bbox;
            float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
            float x1 = static_cast<float>(select_bbox.x1);
            float y1 = static_cast<float>(select_bbox.y1);
            float x2 = static_cast<float>(select_bbox.x2);
            float y2 = static_cast<float>(select_bbox.y2);

            select_idx++;
            for (int32_t i = select_idx; i < num_bbox; i++) {
                if (mask_merged[i] == 1)
                    continue;

                FaceRect& bbox_i = bboxes[i].bbox;
                float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
                float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
                float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
                float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
                if (w <= 0 || h <= 0)
                    continue;

                float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
                float area_intersect = w * h;

                switch (methodType) {
                case 'u':
                    if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                        mask_merged[i] = 1;
                    break;
                case 'm':
                    if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
                        mask_merged[i] = 1;
                    break;
                default:
                    break;
                }
            }
        }
    }
    return bboxes_nms;
}

void MTCNN::Bbox2Square(std::vector<FaceInfo>& bboxes) {
    for(uint32_t i = 0; i < bboxes.size(); i++){
        float w = bboxes[i].bbox.x2 - bboxes[i].bbox.x1 + 1;
        float h = bboxes[i].bbox.y2 - bboxes[i].bbox.y1 + 1;
        float side = std::max<float>(w, h);
        bboxes[i].bbox.x1 += (w - side) * 0.5;
        bboxes[i].bbox.y1 += (h - side) * 0.5;

        bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side - 1);
        bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side - 1);
        bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
        bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

    }
}

std::vector<FaceInfo> MTCNN::BoxRegress(std::vector<FaceInfo>& faceInfo, int stage) {
    std::vector<FaceInfo> bboxes;
    for(uint32_t bboxId = 0; bboxId < faceInfo.size(); bboxId++){
        FaceRect faceRect;
        FaceInfo tempFaceInfo;
        float regw = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1 + 1;
        float regh = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1 + 1;
        faceRect.x1 = faceInfo[bboxId].bbox.x1 + regw * faceInfo[bboxId].regression[0] - 1;
        faceRect.y1 = faceInfo[bboxId].bbox.y1 + regh * faceInfo[bboxId].regression[1] - 1;
        faceRect.x2 = faceInfo[bboxId].bbox.x2 + regw * faceInfo[bboxId].regression[2] - 1;
        faceRect.y2 = faceInfo[bboxId].bbox.y2 + regh * faceInfo[bboxId].regression[3] - 1;
        faceRect.score = faceInfo[bboxId].bbox.score;

        tempFaceInfo.bbox = faceRect;
        tempFaceInfo.regression = faceInfo[bboxId].regression;
        if(stage == 3)
            tempFaceInfo.facePts = faceInfo[bboxId].facePts;
        tempFaceInfo.imgid = faceInfo[bboxId].imgid;
        bboxes.push_back(tempFaceInfo);
    }
    return bboxes;
}

void MTCNN::Padding(const std::vector<cv::Mat>& sample_singles) {
    for(uint32_t i = 0; i < regressed_rects_.size(); i++){
        FaceInfo tempFaceInfo;
        tempFaceInfo = regressed_rects_[i];
        int imgid = tempFaceInfo.imgid;
        int img_w  = sample_singles[imgid].cols;
        int img_h = sample_singles[imgid].rows;
        tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 < 0) ? 0 : regressed_rects_[i].bbox.x1;
        tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 < 0) ? 0 : regressed_rects_[i].bbox.y1;
        tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 > img_w - 1) ? img_w - 1: regressed_rects_[i].bbox.x2;
        tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 > img_h - 1) ? img_h - 1: regressed_rects_[i].bbox.y2;
        regressed_pading_.push_back(tempFaceInfo);
    }
}

void MTCNN::GenerateBoundingBox(__BLOB *confidence, __BLOB *reg, float scale, float thresh, int image_width, int image_height, int imgid) 
{
    int stride = 2;
    int cellSize = 12;

    int width = reg-> width();
    int height = reg-> height();
    int offset = height * width;
    const float* confidence_data = confidence-> cpu_data() + offset;
    //const float* reg_data = reg-> cpu_data();
    const float* reg_data = reg->universe_get_data();

    condidate_rects_.clear();
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int index = y * width + x;
            if(confidence_data[index] >= thresh) {
                float xTop = (int)((y * stride) / scale);
                float yTop = (int)((x * stride) / scale);
                float xBot = (int)((y * stride + cellSize - 1) / scale);
                float yBot = (int)((x * stride + cellSize - 1) / scale);
                FaceRect faceRect;

                faceRect.x1 = yTop;
                faceRect.y1 = xTop;
                faceRect.x2 = yBot;
                faceRect.y2 = xBot;

                faceRect.score  = confidence_data[index];
                FaceInfo faceInfo;
                faceInfo.bbox = faceRect;

                faceInfo.regression = cv::Vec4f(reg_data[offset + index], reg_data[index],
                        reg_data[3 * offset + index], reg_data[2 * offset + index]);
                faceInfo.imgid = imgid;
                condidate_rects_.push_back(faceInfo);
            }
        }
    }
}

MTCNN::MTCNN(const std::string& model_list) {
  std::ifstream infile(model_list.c_str());
  if(!infile){
    std::cerr <<"cannot open file"<<model_list<<std::endl;
    exit(1);
  }
  std::string proto;
  std::string model;
  
  infile >> proto;
  infile >> model;
  PNet_ = new Net<float>(proto, model, TEST);

  infile >> proto;
  infile >> model;
  RNet_ = new Net<float>(proto, model, TEST);

  infile >> proto;
  infile >> model;

  ONet_ = new Net<float>(proto, model, TEST);



  infile.close();
  infile.clear();
}

void MTCNN::WrapInputLayer(std::vector<cv::Mat> *input_channels, __BLOB *input_blob, const int height, const int width) {
    float* input_data = input_blob-> mutable_cpu_data();
    for (int i = 0; i < input_blob-> channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels-> push_back(channel);
        input_data += width * height;
    }
}

void MTCNN::ClassifyFace_MulImage(const std::vector<FaceInfo>& regressed_rects, const std::vector<cv::Mat>& sample_singles,
                                  __NET * net, double thresh, char netName){
    condidate_rects_.clear();
    int numBox = regressed_rects.size();

    std::vector<Datum> datum_vector;
    MemoryDataLayer<float>* mem_data_layer = (MemoryDataLayer<float>*)net-> layers()[0].get();
    int input_width  = mem_data_layer-> width();
    int input_height = mem_data_layer-> height();
    for(int i = 0; i < numBox; i++) {
        int pad_left   = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
        int pad_top    = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
        int pad_right  = std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);
        int pad_bottom = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
        int imgid = regressed_rects[i].imgid;
        cv::Mat crop_img = sample_singles[imgid](cv::Range(regressed_pading_[i].bbox.y1, regressed_pading_[i].bbox.y2 + 1),
                                                 cv::Range(regressed_pading_[i].bbox.x1, regressed_pading_[i].bbox.x2 + 1));
        cv::copyMakeBorder(crop_img, crop_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat resized, res;
        cv::resize(crop_img, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
        resized.convertTo(res, CV_32FC3, 0.0078125, -127.5 * 0.0078125);

        Datum datum;
        CvMatToDatumSignalChannel(res, &datum);
        datum_vector.push_back(datum);
    }
    
    regressed_pading_.clear();
    mem_data_layer-> set_batch_size(numBox);
    mem_data_layer-> AddDatumVector(datum_vector);

    //float no_use_loss = 0;
    //net->Forward(&no_use_loss);
    net->Forward();
    std::string outPutLayerName = (netName == 'r' ? "conv5-2" : "conv6-2");
    std::string pointsLayerName = "conv6-3";

    __BLOB* reg = net-> blob_by_name(outPutLayerName).get();
    __BLOB* confidence = net-> blob_by_name("prob1").get();
    __BLOB* points_offset = net-> blob_by_name(pointsLayerName).get();
    
    const float* confidence_data = confidence-> cpu_data();
    const float* reg_data = reg->universe_get_data();
    const float* points_data;
    //if(netName == 'o') points_data = points_offset-> cpu_data();
    if(netName == 'o') points_data = points_offset->universe_get_data();
    LOG(ERROR)<<(netName == 'o' ? "ONet's" : "RNet's")<<" numBox is "<<numBox;
    for(int i = 0; i < numBox; i++) {
        if(confidence_data[i * 2 + 1] > thresh) {
            FaceRect faceRect;
            faceRect.x1 = regressed_rects[i].bbox.x1;
            faceRect.y1 = regressed_rects[i].bbox.y1;
            faceRect.x2 = regressed_rects[i].bbox.x2;
            faceRect.y2 = regressed_rects[i].bbox.y2 ;
            faceRect.score  = confidence_data[i * 2 + 1];
            FaceInfo faceInfo;
            faceInfo.bbox = faceRect;

            faceInfo.regression = cv::Vec4f(reg_data[4 * i + 1], reg_data[4 * i + 0], reg_data[4 * i + 3], reg_data[4 * i + 2]);

            faceInfo.imgid = regressed_rects[i].imgid;
            if(netName == 'o') {
                FacePts face_pts;
                float w = faceRect.x2 - faceRect.x1 + 1;
                float h = faceRect.y2 - faceRect.y1 + 1;
                for(int j = 0; j < 5; j++){

                    face_pts.x[j] = faceRect.x1 + points_data[2 * j + 10 * i + 1] * w -1;
                    face_pts.y[j] = faceRect.y1 + points_data[2 * j + 10 * i] * h - 1;
                }
                faceInfo.facePts = face_pts;
            }
            condidate_rects_.push_back(faceInfo);
        }
    }
}

bool MTCNN::CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum) {
    if (cv_mat.empty())
        return false;
    int channels = cv_mat.channels();

    datum-> set_channels(cv_mat.channels());
    datum-> set_height(cv_mat.rows);
    datum-> set_width(cv_mat.cols);
    datum-> set_label(0);
    datum-> clear_data();
    datum-> clear_float_data();
    datum-> set_encoded(false);

    int datum_height = datum-> height();
    int datum_width  = datum-> width();
    if(channels == 3){
        for(int c = 0; c < channels; ++c){
            for (int h = 0; h < datum_height; ++h){
                for (int w = 0; w < datum_width; ++w){
                    const float* ptr = cv_mat.ptr<float>(h);
                    datum-> add_float_data(ptr[w * channels + c]);
                }
            }
        }
    }
    return true;
}

void MTCNN::ExtractFeaturesInit(int max_iterations){
  string dir = "lmdb";
  mkdir(dir.c_str(),0777);
  const std::string pnet_data_name = "data";
  const std::string pnet_lmdb_name = dir+"/PNet_data_lmdb";
  const std::string rnet_data_name = "data";
  const std::string rnet_lmdb_name = dir+"/RNet_data_lmdb";
  const std::string onet_data_name = "data";
  const std::string onet_lmdb_name = dir+"/ONet_data_lmdb";
  
  PNet_->ExtractFeaturesInit(pnet_data_name,pnet_lmdb_name,max_iterations);
  RNet_->ExtractFeaturesInit(rnet_data_name,rnet_lmdb_name,max_iterations);
  ONet_->ExtractFeaturesInit(onet_data_name,onet_lmdb_name,max_iterations);
}

bool MTCNN::MultiDetect(const cv::Mat image,std::vector<FaceInfo>& faceInfo, int minSize, double* threshold, double factor,bool bextractFeature) {
  std::vector<cv::Mat> images;
  images.push_back(image);
  bool retp = false, retr = false, reto = false;
  for(uint32_t imgid = 0; imgid < images.size(); imgid++) {
        cv::Mat sample_single,resized;
        images[imgid].copyTo(sample_single);
        int width  = sample_single.cols;
        int height = sample_single.rows;
        int minWH = std::min(height, width);
        int factor_count = 0;
        double m = 12. / minSize;
        minWH *= m;
        std::vector<double> scales;
        while (minWH >= 12) {
            scales.push_back(m * std::pow(factor, factor_count));
            minWH *= factor;
            ++factor_count;
        }
        /* PNet */
        __BLOB* input_blob = PNet_->blob_by_name("data").get();
        
        
        for(int i = 0; i < factor_count; i++)
        {
            double scale = scales[i];
            int ws = std::ceil(width * scale);
            int hs = std::ceil(height * scale);
            cv::resize(sample_single, resized, cv::Size(ws, hs), 0, 0, cv::INTER_NEAREST);
           
           
            resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5 * 0.0078125);           
    
            std::vector<cv::Mat> input_channels;

            input_blob->Reshape(1, 3, hs, ws);
            
            std::cout<<"the "<<imgid<<" th image "<<" current  scale "<<scale<<" current ws "<<ws<<" current hs "<<hs<<std::endl;
            input_blob->universe_fill_data(resized);
                        
            PNet_->Forward();
            if(bextractFeature){
              retp = PNet_->ExtractFeatures();
            }
            __BLOB* reg = PNet_-> blob_by_name("conv4-2").get();
            __BLOB* confidence = PNet_-> blob_by_name("prob1").get();            

            GenerateBoundingBox(confidence, reg, scale, threshold[0], ws, hs, imgid);
            std::vector<FaceInfo> bboxes_nms = MultiNMS(condidate_rects_,0.5, 'u', 1);
            total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
                         
            
        }
           
    }
  //Ufw::set_mode(Ufw::FP32);
    int numBox = total_boxes_.size();
    std::cout<<"The total numBox is "<<numBox<<std::endl;
    if(numBox != 0) {
        total_boxes_ = MultiNMS(total_boxes_, 0.7, 'u', images.size());
        regressed_rects_ = BoxRegress(total_boxes_, 1);
        total_boxes_.clear();
        Bbox2Square(regressed_rects_);
        Padding(images);
        /* RNet */
        ClassifyFace_MulImage(regressed_rects_, images, RNet_, threshold[1], 'r');
        if(bextractFeature){
          retr = RNet_->ExtractFeatures();
        }
        condidate_rects_ = MultiNMS(condidate_rects_, 0.7, 'u', images.size());
        regressed_rects_ = BoxRegress(condidate_rects_, 2);
        Bbox2Square(regressed_rects_);
        Padding(images);

        numBox = regressed_rects_.size();
        if(numBox != 0) {
            /* ONet */        
            ClassifyFace_MulImage(regressed_rects_, images,ONet_, threshold[2], 'o');
            if(bextractFeature){
              reto = ONet_->ExtractFeatures();
            }
            regressed_rects_ = BoxRegress(condidate_rects_, 3);
            faceInfo = MultiNMS(regressed_rects_, 0.7, 'm', images.size());
            
        }
    }
    regressed_pading_.clear();
    regressed_rects_.clear();
    condidate_rects_.clear();
    return retp & retr & reto;
}

void detector_init(FaceDetector* face_detector, const std::string& model_list){
    MTCNN* detector = new MTCNN(model_list);
    *face_detector = (FaceDetector)detector;
}
    
void detector_detect(FaceDetector face_detector, const cv::Mat image, std::vector<FaceInfo>& faceInfo, int minSize, double* threshold, double factor){
    MTCNN* detector = (MTCNN*)face_detector;
    detector->MultiDetect(image, faceInfo, minSize, threshold, factor);
}

void detector_dump_init(FaceDetector* face_detector, const std::string& model_list, int max_iterations){
  MTCNN* detector = new MTCNN(model_list);
  detector->ExtractFeaturesInit(max_iterations);
  *face_detector = (FaceDetector)detector;
}

bool detector_dump(FaceDetector face_detector, const cv::Mat image, std::vector<FaceInfo>& faceInfo, int minSize, double* threshold, double factor){
    MTCNN* detector = (MTCNN*)face_detector;
    bool ret =  detector->MultiDetect(image, faceInfo, minSize, threshold, factor,true);
    return ret;
}

void detector_destroy(FaceDetector face_detector){
    MTCNN* detector = (MTCNN*)face_detector;
    delete detector;
}
