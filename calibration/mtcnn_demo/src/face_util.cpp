#include "face.hpp"

//#include <cio>
int _match_spec(const char* spec, const char* text) {
    /*
     * If the whole specification string was consumed and
     * the input text is also exhausted: it's a match.
     */
    if (spec[0] == '\0' && text[0] == '\0') {
        return 1;
    }

    /* A star matches 0 or more characters. */
    if (spec[0] == '*') {
        /*
         * Skip the star and try to find a match after it
         * by successively incrementing the text pointer.
         */
        do {
            if (_match_spec(spec + 1, text)) {
                return 1;
            }
        } while (*text++ != '\0');
    }

    /*
     * An interrogation mark matches any character. Other
     * characters match themself. Also, if the input text
     * is exhausted but the specification isn't, there is
     * no match.
     */
    if (text[0] != '\0' && (spec[0] == '?' || spec[0] == text[0])) {
        return _match_spec(spec + 1, text + 1);
    }

    return 0;
}

int match_spec(const char* spec, const char* text) {
    /* On Windows, *.* matches everything. */
    if (strcmp(spec, "*.*") == 0) {
        return 1;
    }

    return _match_spec(spec, text);
}

#define _A_NORMAL   0x00    /* Normal file.     */
#define _A_RDONLY   0x01    /* Read only file.  */
#define _A_HIDDEN   0x02    /* Hidden file.     */
#define _A_SYSTEM   0x04    /* System file.     */
#define _A_SUBDIR   0x10    /* Subdirectory.    */
#define _A_ARCH     0x20    /* Archive file.    */

struct _finddata_t {
    unsigned attrib;
    time_t time_create;
    time_t time_access;
    time_t time_write;
    off_t size;
    char name[260];
};

/*
 * Returns a unique search handle identifying the file or group of
 * files matching the filespec specification, which can be used in
 * a subsequent call to findnext or to findclose. Otherwise, findfirst
 * returns NULL and sets errno to EINVAL if filespec or fileinfo
 * was NULL or if the operating system returned an unexpected error
 * and ENOENT if the file specification could not be matched.
 */
intptr_t _findfirst(const char* filespec, struct _finddata_t* fileinfo);

/*
 * Find the next entry, if any, that matches the filespec argument
 * of a previous call to findfirst, and then alter the fileinfo
 * structure contents accordingly. If successful, returns 0. Otherwise,
 * returns -1 and sets errno to EINVAL if handle or fileinfo was NULL
 * or if the operating system returned an unexpected error and ENOENT
 * if no more matching files could be found.
 */
int _findnext(intptr_t handle, struct _finddata_t* fileinfo);

/*
 * Closes the specified search handle and releases associated
 * resources. If successful, findclose returns 0. Otherwise, it
 * returns -1 and sets errno to ENOENT, indicating that no more
 * matching files could be found.
 */
int _findclose(intptr_t handle);

#ifdef __linux__
#define _XOPEN_SOURCE 700   /* SUSv4 */
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <libgen.h>
#include <limits.h>
#include <dirent.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>

#ifdef __linux__
#include <alloca.h>
#endif

//#include "findfirst.h"
//#include "spec.h"

#define DOTDOT_HANDLE    0L
#define INVALID_HANDLE  -1L

typedef struct fhandle_t {
    DIR* dstream;
    short dironly;
    char* spec;
} fhandle_t;

static void fill_finddata(struct stat* st, const char* name,
        struct _finddata_t* fileinfo);

static intptr_t findfirst_dotdot(const char* filespec,
        struct _finddata_t* fileinfo);

static intptr_t findfirst_in_directory(const char* dirpath,
        const char* spec, struct _finddata_t* fileinfo);

static void findfirst_set_errno();

intptr_t _findfirst(const char* filespec, struct _finddata_t* fileinfo) {
    const char* rmslash;      /* Rightmost forward slash in filespec. */
    const char* spec;   /* Specification string. */

    if (!fileinfo || !filespec) {
        errno = EINVAL;
        return INVALID_HANDLE;
    }

    if (filespec[0] == '\0') {
        errno = ENOENT;
        return INVALID_HANDLE;
    }

    rmslash = strrchr(filespec, '/');

    if (rmslash != NULL) {
        /*
         * At least one forward slash was found in the filespec
         * string, and rmslash points to the rightmost one. The
         * specification part, if any, begins right after it.
         */
        spec = rmslash + 1;
    } else {
        spec = filespec;
    }

    if (strcmp(spec, ".") == 0 || strcmp(spec, "..") == 0) {
        /* On Windows, . and .. must return canonicalized names. */
        return findfirst_dotdot(filespec, fileinfo);
    } else if (rmslash == filespec) {
        /*
         * Since the rightmost slash is the first character, we're
         * looking for something located at the file system's root.
         */
        return findfirst_in_directory("/", spec, fileinfo);
    } else if (rmslash != NULL) {
        /*
         * Since the rightmost slash isn't the first one, we're
         * looking for something located in a specific folder. In
         * order to open this folder, we split the folder path from
         * the specification part by overwriting the rightmost
         * forward slash.
         */
        size_t pathlen = strlen(filespec) +1;
        char* dirpath = (char*)alloca(pathlen);
        memcpy(dirpath, filespec, pathlen);
        dirpath[rmslash - filespec] = '\0';
        return findfirst_in_directory(dirpath, spec, fileinfo);
    } else {
        /*
         * Since the filespec doesn't contain any forward slash,
         * we're looking for something located in the current
         * directory.
         */
        return findfirst_in_directory(".", spec, fileinfo);
    }
}

/* Perfom a scan in the directory identified by dirpath. */
static intptr_t findfirst_in_directory(const char* dirpath,
        const char* spec, struct _finddata_t* fileinfo) {
    DIR* dstream;
    fhandle_t* ffhandle;

    if (spec[0] == '\0') {
        errno = ENOENT;
        return INVALID_HANDLE;
    }

    if ((dstream = opendir(dirpath)) == NULL) {
        findfirst_set_errno();
        return INVALID_HANDLE;
    }

    if ((ffhandle = (fhandle_t *)malloc(sizeof(fhandle_t))) == NULL) {
        closedir(dstream);
        errno = ENOMEM;
        return INVALID_HANDLE;
    }

    /* On Windows, *. returns only directories. */
    ffhandle->dironly = strcmp(spec, "*.") == 0 ? 1 : 0;
    ffhandle->dstream = dstream;
    ffhandle->spec = strdup(spec);

    if (_findnext((intptr_t) ffhandle, fileinfo) != 0) {
        _findclose((intptr_t) ffhandle);
        errno = ENOENT;
        return INVALID_HANDLE;
    }

    return (intptr_t) ffhandle;
}

/* On Windows, . and .. return canonicalized directory names. */
static intptr_t findfirst_dotdot(const char* filespec,
        struct _finddata_t* fileinfo) {
    char* dirname;
    char* canonicalized;
    struct stat st;

    if (stat(filespec, &st) != 0) {
        findfirst_set_errno();
        return INVALID_HANDLE;
    }

    /* Resolve filespec to an absolute path. */
    if ((canonicalized = realpath(filespec, NULL)) == NULL) {
        findfirst_set_errno();
        return INVALID_HANDLE;
    }

    /* Retrieve the basename from it. */
    dirname = basename(canonicalized);

    /* Make sure that we actually have a basename. */
    if (dirname[0] == '\0') {
        free(canonicalized);
        errno = ENOENT;
        return INVALID_HANDLE;
    }

    /* Make sure that we won't overflow finddata_t::name. */
    if (strlen(dirname) > 259) {
        free(canonicalized);
        errno = ENOMEM;
        return INVALID_HANDLE;
    }

    fill_finddata(&st, dirname, fileinfo);

    free(canonicalized);
   return DOTDOT_HANDLE;
}

/*
 * Windows implementation of _findfirst either returns EINVAL,
 * ENOENT or ENOMEM. This function makes sure that the above
 * implementation doesn't return anything else when an error
 * condition is encountered.
 */
static void findfirst_set_errno() {
    if (errno != ENOENT &&
        errno != ENOMEM &&
        errno != EINVAL) {
        errno = EINVAL;
    }
}

static void fill_finddata(struct stat* st, const char* name,
        struct _finddata_t* fileinfo) {
    fileinfo->attrib = S_ISDIR(st->st_mode) ? _A_SUBDIR : _A_NORMAL;
    fileinfo->size = st->st_size;
    fileinfo->time_create = st->st_ctime;
    fileinfo->time_access = st->st_atime;
    fileinfo->time_write = st->st_mtime;
    strcpy(fileinfo->name, name);
}

int _findnext(intptr_t fhandle, struct _finddata_t* fileinfo) {
    struct dirent entry, *result;
    struct fhandle_t* handle;
    struct stat st;

    if (fhandle == DOTDOT_HANDLE) {
        errno = ENOENT;
        return -1;
    }

    if (fhandle == INVALID_HANDLE || !fileinfo) {
        errno = EINVAL;
        return -1;
    }

    handle = (struct fhandle_t*) fhandle;

    while (readdir_r(handle->dstream, &entry, &result) == 0 && result != NULL) {
        if (!handle->dironly && !match_spec(handle->spec, entry.d_name)) {
            continue;
        }

        if (fstatat(dirfd(handle->dstream), entry.d_name, &st, 0) == -1) {
            return -1;
        }

        if (handle->dironly && !S_ISDIR(st.st_mode)) {
            continue;
        }

        fill_finddata(&st, entry.d_name, fileinfo);

        return 0;
    }

    errno = ENOENT;
    return -1;
}

int _findclose(intptr_t fhandle) {
    struct fhandle_t* handle;

    if (fhandle == DOTDOT_HANDLE) {
        return 0;
    }

    if (fhandle == INVALID_HANDLE) {
        errno = ENOENT;
        return -1;
    }

    handle = (struct fhandle_t*) fhandle;

    closedir(handle->dstream);
    free(handle->spec);
    free(handle);

    return 0;
}



/////////
void getFiles(std::string path, std::vector<std::string>& files )  
{  
    //文件句柄  
    long   hFile   =   0;  
    //文件信息  
    struct _finddata_t fileinfo;  
    std::string p;
    //std::cout << p.assign(path) << std::endl;
    if((hFile = _findfirst(p.assign(path).append("/*").c_str(),&fileinfo)) !=  -1)  
    {  
        do  
        {  
            //如果是目录,迭代之  
            //如果不是,加入列表  
            //if((fileinfo.attrib &  _A_SUBDIR))  
            //{  
            //    if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
            //        getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
            //}  
            //else  
            //{ 
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0 && fileinfo.name[0] != '.') {
                  files.push_back(p.assign(path).append("/").append(fileinfo.name) );  
                }
              
            //}  
        }while(_findnext(hFile, &fileinfo)  == 0);  
        _findclose(hFile);  
    }
}

void getNames(std::string path, std::vector<std::string>& names )
{
  long   hFile   =   0;
  struct _finddata_t fileinfo;
  std::string p;
  if((hFile = _findfirst(p.assign(path).append("/*").c_str(),&fileinfo)) !=  -1)
  {
    do {
      //files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
      if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0 && fileinfo.name[0] != '.') {
        names.push_back(std::string(fileinfo.name));
      }
    } while(_findnext(hFile, &fileinfo)  == 0);
    _findclose(hFile);
  }
}    

void getFaces(std::vector<std::string>& files, std::vector<std::vector<std::string> >& faces)
{
  for (unsigned int i = 0; i < files.size(); i++) {
    std::vector<std::string> temp;
    getFiles(files[i], temp);
    faces.push_back(temp);
  }
}

///

inline float calc_sqrt(const std::vector<float>& feature)
{
  float a = 0;
  for (unsigned int i = 0 ; i < feature.size(); i++) {
    a += feature[i] * feature[i];
  }
  return sqrt(a);
}


//inline float calc_cosine(const std::vector<float>& feature1, const std::vector<float>& feature2)
float calc_cosine(const std::vector<float>& feature1, const std::vector<float>& feature2)
{
  float a = 0;
  for (unsigned int i = 0 ; i < feature1.size(); i++) {
    a += feature1[i] * feature2[i];
  }
  float b = calc_sqrt(feature1) * calc_sqrt(feature2);
  //return a / (b + 0.000000001);
  return a / b;
}

inline float calc_euclidean(const std::vector<float>& feature1, const std::vector<float>& feature2)
{
  float a = 0;
  for (unsigned int i = 0; i < feature1.size(); i++) {
    a += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i]);
  }
  return -sqrt(a);
}

inline float calc_dot(const std::vector<float>& feature1, const std::vector<float>& feature2)
{
  float a = 0;
  for (unsigned int i = 0 ; i < feature1.size(); i++) {
    a += feature1[i] * feature2[i];
  }
  return a;
}

inline int find_person(const std::vector<float>& feature, const std::vector<std::vector<float> >&allfeatures, std::vector<std::pair<int,float> >& sims)
{
  for (unsigned int i = 0; i < allfeatures.size(); i++) {
    sims.push_back(std::make_pair(i,calc_cosine(feature,allfeatures[i])));
    //sims.push_back(std::make_pair(i,calc_euclidean(feature,allfeatures[i])));
  }
  int index = 0;
  float sim = 0;
  for (unsigned int i = 0; i < sims.size(); i++) {
    if (sims[i].second > sim) {
      sim = sims[i].second;
      index = i;
    }
  }
  return index;
}

extern "C" 
{
#include <pmmintrin.h>
#include <immintrin.h>   // (Meta-header)
}

extern "C"
{
#include <emmintrin.h>
#include <mmintrin.h>
}

extern "C" 
{
#include <smmintrin.h>
}

#if defined(_MSC_VER)
#define ALIGNED_(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define ALIGNED_(x) __attribute__ ((aligned(x)))
#endif
#endif

float sse_inner(const float* a, const float* b, unsigned int size)
{
        float z = 0.0f, fres = 0.0f;
        //__declspec(align(16)) float ftmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        //__attribute__ ((aligned(16))) float ftmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        ALIGNED_(16) float ftmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        __m128 mres;

        if ((size / 4) != 0) {
                mres = _mm_load_ss(&z);
                for (unsigned int i = 0; i < size / 4; i++)
                        mres = _mm_add_ps(mres, _mm_mul_ps(_mm_loadu_ps(&a[4*i]),
                        _mm_loadu_ps(&b[4*i])));

                //mres = a,b,c,d
                __m128 mv1 = _mm_movelh_ps(mres, mres);     //a,b,a,b
                __m128 mv2 = _mm_movehl_ps(mres, mres);     //c,d,c,d
                mres = _mm_add_ps(mv1, mv2);                //res[0],res[1]

                _mm_store_ps(ftmp, mres);                

                fres = ftmp[0] + ftmp[1];
        }

        if ((size % 4) != 0) {
                for (unsigned int i = size - size % 4; i < size; i++)
                        fres += a[i] * b[i];
        }

        return fres;
}
#if 0
float sse3_inner(const float* a, const float* b, unsigned int size)
{
        float z = 0.0f, fres = 0.0f;
        
        if ((size / 4) != 0) {
                const float* pa = a;
                const float* pb = b;
                __asm (
                        //"movss   xmm0, xmmword ptr[z]"
                        "movss xmm0, XMMWORD PTR[z]"
                );
                for (unsigned int i = 0; i < size / 4; i++) {
                        __asm ( 
                                //"mov     eax, dword ptr[pa]"
                                "mov     eax, DWORD PTR [pa]"
                                //"mov     ebx, dword ptr[pb]"
                                "mov     ebx, DWORD PTR [pb]"
                                "movups  xmm1, [eax]"
                                "movups  xmm2, [ebx]"
                                "mulps   xmm1, xmm2"
                                "addps   xmm0, xmm1"
                                
                        );
                        pa += 4;
                        pb += 4;
                }  
                __asm (
                        "haddps  xmm0, xmm0"
                        "haddps  xmm0, xmm0"
                        "movss   dword ptr[fres], xmm0"                      
                );               
        }

        return fres;
}
#endif

inline float calc_dot_sse(const std::vector<float>& feature1, const std::vector<float>& feature2)
{
  const float *a = &(feature1[0]);
  const float *b = &(feature2[0]);
  return sse_inner(a, b, feature1.size());
}
 
cv::Mat tformfwd(const cv::Mat& trans, const cv::Mat& uv){
  cv::Mat uv_h = cv::Mat::ones(uv.rows, 3, CV_64FC1);
  uv.copyTo(uv_h(cv::Rect(0, 0, 2, uv.rows)));
  cv::Mat xv_h = uv_h*trans;
  return xv_h(cv::Rect(0, 0, 2, uv.rows));
}

cv::Mat find_none_flectives_similarity(const cv::Mat& uv, const cv::Mat& xy){
  cv::Mat A = cv::Mat::zeros(2*xy.rows, 4, CV_64FC1);
  cv::Mat b = cv::Mat::zeros(2*xy.rows, 1, CV_64FC1);
  cv::Mat x = cv::Mat::zeros(4, 1, CV_64FC1);

  xy(cv::Rect(0, 0, 1, xy.rows)).copyTo(A(cv::Rect(0, 0, 1, xy.rows)));//x
  xy(cv::Rect(1, 0, 1, xy.rows)).copyTo(A(cv::Rect(1, 0, 1, xy.rows)) );//y
  A(cv::Rect(2, 0, 1, xy.rows)).setTo(1.);

  xy(cv::Rect(1, 0, 1, xy.rows)).copyTo(A(cv::Rect(0, xy.rows, 1, xy.rows)));//y
  (xy(cv::Rect(0, 0, 1, xy.rows))).copyTo(A(cv::Rect(1, xy.rows, 1, xy.rows)));//-x
  A(cv::Rect(1, xy.rows, 1, xy.rows)) *= -1;
  A(cv::Rect(3, xy.rows, 1, xy.rows)).setTo(1.);

  uv(cv::Rect(0, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, 0, 1, uv.rows)));
  uv(cv::Rect(1, 0, 1, uv.rows)).copyTo(b(cv::Rect(0, uv.rows, 1, uv.rows)));

  cv::solve(A, b, x, cv::DECOMP_SVD);
  cv::Mat trans_inv = (cv::Mat_<double>(3, 3) << x.at<double>(0),  -x.at<double>(1), 0,
          x.at<double>(1),  x.at<double>(0), 0,
          x.at<double>(2),  x.at<double>(3), 1);
  cv::Mat trans = trans_inv.inv(cv::DECOMP_SVD);
  trans.at<double>(0, 2) = 0;
  trans.at<double>(1, 2) = 0;
  trans.at<double>(2, 2) = 1;

  return trans;
}

cv::Mat find_similarity(const cv::Mat& uv, const cv::Mat& xy){
  cv::Mat trans1 =find_none_flectives_similarity(uv, xy);
  cv::Mat xy_reflect = xy;
  xy_reflect(cv::Rect(0, 0, 1, xy.rows)) *= -1;
  cv::Mat trans2r = find_none_flectives_similarity(uv, xy_reflect);
  cv::Mat reflect = (cv::Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

  cv::Mat trans2 = trans2r*reflect;
  cv::Mat xy1 = tformfwd(trans1, uv);
  double norm1 = cv::norm(xy1 - xy);

  cv::Mat xy2 = tformfwd(trans2, uv);
  double norm2 = cv::norm(xy2 - xy);

  cv::Mat trans;
  if(norm1 < norm2){
      trans = trans1;
  } else {
      trans = trans2;
  }
  return trans;
}

cv::Mat get_similarity_transform(const std::vector<cv::Point2f>& src_points, const std::vector<cv::Point2f>& dst_points, bool reflective = true){
  cv::Mat trans;
  cv::Mat src((int)src_points.size(), 2, CV_32FC1, (void*)(&src_points[0].x));
  src.convertTo(src, CV_64FC1);

  cv::Mat dst((int)dst_points.size(), 2, CV_32FC1, (void*)(&dst_points[0].x));
  dst.convertTo(dst, CV_64FC1);

  if(reflective){
      trans = find_similarity(src, dst);
  } else {
      trans = find_none_flectives_similarity(src, dst);
  }
  cv::Mat trans_cv2 = trans(cv::Rect(0, 0, 2, trans.rows)).t();

  return trans_cv2;
}

cv::Mat align_face(const cv::Mat& src, const FaceInfo faceInfo, int width, int height)
{ 
  const int ReferenceWidth = 96;
  const int ReferenceHeight = 112;
  std::vector<cv::Point2f> detect_points;
  for (int j = 0; j < 5; ++j) {
    cv::Point2f e;
    e.x = faceInfo.facePts.x[j];
    e.y = faceInfo.facePts.y[j];
    detect_points.push_back(e);
  }
  std::vector<cv::Point2f> reference_points;
  reference_points.push_back(
    cv::Point2f(30.29459953,  51.69630051));
  reference_points.push_back(
    cv::Point2f(65.53179932,  51.50139999));
  reference_points.push_back(
    cv::Point2f(48.02519989,  71.73660278));
  reference_points.push_back(
    cv::Point2f(33.54930115,  92.36550140));
  reference_points.push_back(
    cv::Point2f(62.72990036,  92.20410156));
  for (int j = 0;j < 5; ++j) {
    reference_points[j].x += (width - ReferenceWidth)/2.0f;
    reference_points[j].y += (height - ReferenceHeight)/2.0f;
  }
  cv::Mat tfm = get_similarity_transform(detect_points, reference_points);
  cv::Mat aligned_face;
  std::cout<<"width:"<<width<<" height:"<<height<<std::endl;
  std::cout<<"src.cols:"<<src.cols<<" src.rows:"<<src.rows<<std::endl;
  cv::warpAffine(src, aligned_face, tfm, cv::Size(width, height));
  return aligned_face;
}

bool compareArea(const FaceInfo & a, const FaceInfo & b) {
  return (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1) > (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1);
}

bool compareDis(const FaceInfo & a, const FaceInfo & b) {
  return a.distance < b.distance;
}

bool compareImgId(const FaceInfo & a, const FaceInfo & b) {
  return a.imgid < b.imgid;
}
