#include <stdio.h>
#include <stdlib.h>
#include "face.hpp"
#include <ufw/ufw.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <libgen.h>
#include <sys/stat.h>
#include "boost/algorithm/string.hpp"


using namespace ufw;
using namespace std;


DEFINE_string(mode, "",
    "Optional; mode type for inference (FLOAT, INT8), "
    "separated by ','.");
DEFINE_string(model_list, "", "model file list: \
pnet.prototxt,pnet.caffemodel, \
rnet.prototxt,rnet.caffemodel, \
onet.prototxt,onet.caffemodel");
DEFINE_string(image_list, "", "input file list.");
DEFINE_string(fddb_dir, "", "dir of fddb dataset.");
DEFINE_int32(iterations, 1000,"The number of iterations to run for dumping lmdb dataset.");
DEFINE_bool(show, false,"show window");

double threshold[3] = {0.5, 0.7, 0.7};
double factor = 0.89;
int minSize = 40;

// A simple registry for ufw commands.
typedef int (*BrewFunction)();
typedef std::map<ufw::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const ufw::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available ufw actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

void read_to_vector(const string image_list, vector<string>& imageArray)
{
  ifstream infile(image_list.c_str());
  if(!infile){
    std::cout <<"cannot open file"<<image_list<<std::endl;
    exit(1); 
  }
  string imageName;
  while(infile >> imageName){
    imageArray.push_back(imageName);
  }
  infile.close();
  std::cout<<"The total num of image file is "<<imageArray.size()<<std::endl;
}

int demo()
{
  FLAGS_alsologtostderr = 1;
  
  //init detector
  FaceDetector detector;
  detector_init(&detector, FLAGS_model_list);

  vector<string> imageArray;
  read_to_vector(FLAGS_image_list, imageArray);

  for(uint32_t i = 0; i < imageArray.size(); i++){
    const string imageName = imageArray[i];
    cv::Mat image = cv::imread(imageName.c_str());
    image = image.t();
    
    std::vector<FaceInfo> faceInfos; //store detection result
    detector_detect(detector, image, faceInfos, minSize, threshold, factor);

    if(faceInfos.size() > 0){
      std::cout << "detected "<<faceInfos.size()<< " faces in image: " << imageName << std::endl;

      //draw rectangles
      for (uint32_t k = 0; k < faceInfos.size(); k++){
        cv::Rect rc;
        rc.x = faceInfos[k].bbox.x1;
        rc.y = faceInfos[k].bbox.y1;
        rc.width = faceInfos[k].bbox.x2 - faceInfos[k].bbox.x1;
        rc.height = faceInfos[k].bbox.y2 - faceInfos[k].bbox.y1;
        std::cout << rc.x << " " << rc.y << " " << rc.width << " " << rc.height << std::endl;
        cv::rectangle(image, rc, cv::Scalar(255,0,255), 2, 1, 0);
      }
      
      if(FLAGS_show){
        cv::namedWindow(imageName, CV_WINDOW_AUTOSIZE);
        cv::imshow(imageName, image.t());
        cv::waitKey(25);
      }else{
        int idx = imageName.find_last_of("/");
        string substr = imageName.substr(idx+1);
        idx = substr.find_last_of(".");
        substr = substr.substr(0,idx);
        mkdir("results",0777);
        string resultImageName;
        if(FLAGS_mode=="INT8"){
          resultImageName = "results/"+substr+"_int8.png";
          cv::imwrite(resultImageName, image.t());
        }else{
          resultImageName = "results/"+substr+".png";
          cv::imwrite(resultImageName, image.t());
        }
        LOG(INFO)<<resultImageName;
      }
    }else{
      std::cout << "nothing detected in image " << i << std::endl;
    }
  }

  if(FLAGS_show){
    std::cout << "wait any key pressed." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  // gc
  detector_destroy(detector);
  
  std::cout << "done." << std::endl;
  return 0;
}
RegisterBrewFunction(demo);

int fddb()
{
  //fddb dataset 
  vector<string> imageArray;
  string outfileName = "data/image_file_list_fddb.txt";
  std::ofstream outfile(outfileName.c_str());
  for(int i=0;i<10;i++){
    string fileName(FLAGS_fddb_dir+"/FDDB-folds/FDDB-fold-");
    char postFixName[50];
    sprintf(postFixName,"%02d.txt",i+1);
    fileName = fileName+postFixName;
    std::cout<<fileName<<std::endl;
    
    std::ifstream infile(fileName.c_str());
    if(!infile){
      std::cout <<"cannot open file"<<fileName<<std::endl;
      return -1; 
    }

    string imageName;
    string preFix(FLAGS_fddb_dir+"/originalPics/");
    string postFix(".jpg");
    while(infile >> imageName){
      imageName = preFix + imageName+ postFix;
      outfile<<imageName<<std::endl;
    }
    infile.close();
  }
  
  outfile.close();
  return 0;
}
RegisterBrewFunction(fddb);

int dump()
{
  FLAGS_alsologtostderr = 1;
  Ufw::set_mode(Ufw::FP32);

  vector<string> imageArray;
  read_to_vector(FLAGS_image_list, imageArray);

  //init detector
  FaceDetector detector;
  int max_iterations = FLAGS_iterations;
  max_iterations = max_iterations < imageArray.size() ? max_iterations : imageArray.size();
  detector_dump_init(&detector, FLAGS_model_list,max_iterations);

  for(uint32_t i = 0; i < imageArray.size(); i++){
    const string imageName = imageArray[i];
    cv::Mat image = cv::imread(imageName.c_str());
    image = image.t();
    
    std::vector<FaceInfo> faceInfos; //store detection result
    bool ret = detector_dump(detector, image, faceInfos, minSize, threshold, factor);
    if(ret){
      break;
    }
  }
  
  // gc
  detector_destroy(detector);
  
  std::cout << "done." << std::endl;
  return 0;
}
RegisterBrewFunction(dump);
int main(int argc, char** argv) {
  gflags::SetUsageMessage("command line brew\n"
      "usage: mtcnn <command> <args>\n\n"
      "commands:\n"
      "  demo        show images with rectangle\n"
      "  fddb        get fddb file list\n"
      "  dump        dump input date to lmdb\n");
  //Run tool or show usage.
  ufw::GlobalInit(&argc, &argv);
  
  if(FLAGS_mode=="INT8"){
    Ufw::set_mode(Ufw::INT8);
  }else{
    Ufw::set_mode(Ufw::FP32);  
  }
  LOG(ERROR) << "Inference mode:"<<Ufw::mode();  
  
  if (argc == 2) {
    return GetBrewFunction(std::string(argv[1]))();
  }else{
    gflags::ShowUsageWithFlagsRestrict(argv[0], "demo");
  }
}
