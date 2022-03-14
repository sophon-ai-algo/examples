//
// Created by yuan on 3/16/21.
//

#include "openpose.h"
#include "pose_postprocess.h"

#define POSE_COCO_COLORS_RENDER_GPU \
	255.f, 0.f, 85.f, \
	255.f, 0.f, 0.f, \
	255.f, 85.f, 0.f, \
	255.f, 170.f, 0.f, \
	255.f, 255.f, 0.f, \
	170.f, 255.f, 0.f, \
	85.f, 255.f, 0.f, \
	0.f, 255.f, 0.f, \
	0.f, 255.f, 85.f, \
	0.f, 255.f, 170.f, \
	0.f, 255.f, 255.f, \
	0.f, 170.f, 255.f, \
	0.f, 85.f, 255.f, \
	0.f, 0.f, 255.f, \
	255.f, 0.f, 170.f, \
	170.f, 0.f, 255.f, \
	255.f, 0.f, 255.f, \
	85.f, 0.f, 255.f

const std::vector<float> POSE_COCO_COLORS_RENDER{ POSE_COCO_COLORS_RENDER_GPU };
const std::vector<unsigned int> POSE_COCO_PAIRS_RENDER{1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17};
const unsigned int POSE_MAX_PEOPLE = 96;

// Round functions
// Signed
template<typename T>
inline char charRound(const T a)
{
    return char(a+0.5f);
}

template<typename T>
inline signed char sCharRound(const T a)
{
    return (signed char)(a+0.5f);
}

template<typename T>
inline int intRound(const T a)
{
    return int(a+0.5f);
}

template<typename T>
inline long longRound(const T a)
{
    return long(a+0.5f);
}

template<typename T>
inline long long longLongRound(const T a)
{
    return (long long)(a+0.5f);
}

// Unsigned
template<typename T>
inline unsigned char uCharRound(const T a)
{
    return (unsigned char)(a+0.5f);
}

template<typename T>
inline unsigned int uIntRound(const T a)
{
    return (unsigned int)(a+0.5f);
}

template<typename T>
inline unsigned long ulongRound(const T a)
{
    return (unsigned long)(a+0.5f);
}

template<typename T>
inline unsigned long long uLongLongRound(const T a)
{
    return (unsigned long long)(a+0.5f);
}

// Max/min functions
template<typename T>
inline T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

template<class T>
inline T fastTruncate(T value, T min = 0, T max = 1)
{
    return fastMin(max, fastMax(min, value));
}


OpenPosePostProcess::OpenPosePostProcess() {

}

OpenPosePostProcess::~OpenPosePostProcess() {

}


int OpenPosePostProcess::Nms(PoseBlobPtr bottom_blob, PoseBlobPtr top_blob, float threshold)
{
    //maxPeaks就是最大人数，+1是为了第一位存个数
    //算法，是每个点，如果大于阈值，同时大于上下左右值的时候，则认为是峰值

    //算法很简单，featuremap的任意一个点，其上下左右和斜上下左右，都小于自身，就认为是要的点
    //然后以该点区域，选择7*7区域，按照得分值和x、y来计算最合适的亚像素坐标

    int w = bottom_blob->width();
    int h = bottom_blob->height();
    int plane_offset = w * h;
    float* ptr = bottom_blob->data();
    float* top_ptr = top_blob->data();
    int top_plane_offset = top_blob->width() * top_blob->height();
    int max_peaks = top_blob->height() - 1;

    for (int n = 0; n < bottom_blob->num(); ++n){
        for (int c = 0; c < bottom_blob->channels() - 1; ++c){

            int num_peaks = 0;
            for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y){
                for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x){
                    float value = ptr[y*w + x];
                    if (value > threshold){
                        const float topLeft = ptr[(y - 1)*w + x - 1];
                        const float top = ptr[(y - 1)*w + x];
                        const float topRight = ptr[(y - 1)*w + x + 1];
                        const float left = ptr[y*w + x - 1];
                        const float right = ptr[y*w + x + 1];
                        const float bottomLeft = ptr[(y + 1)*w + x - 1];
                        const float bottom = ptr[(y + 1)*w + x];
                        const float bottomRight = ptr[(y + 1)*w + x + 1];

                        if (value > topLeft && value > top && value > topRight
                            && value > left && value > right
                            && value > bottomLeft && value > bottom && value > bottomRight)
                        {
                            //计算亚像素坐标
                            float xAcc = 0;
                            float yAcc = 0;
                            float scoreAcc = 0;
                            for (int kx = -3; kx <= 3; ++kx){
                                int ux = x + kx;
                                if (ux >= 0 && ux < w){
                                    for (int ky = -3; ky <= 3; ++ky){
                                        int uy = y + ky;
                                        if (uy >= 0 && uy < h){
                                            float score = ptr[uy * w + ux];
                                            xAcc += ux * score;
                                            yAcc += uy * score;
                                            scoreAcc += score;
                                        }
                                    }
                                }
                            }

                            xAcc /= scoreAcc;
                            yAcc /= scoreAcc;
                            scoreAcc = value;
                            top_ptr[(num_peaks + 1) * 3 + 0] = xAcc;
                            top_ptr[(num_peaks + 1) * 3 + 1] = yAcc;
                            top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
                            num_peaks++;
                        }
                    }
                }
            }
            top_ptr[0] = num_peaks;
            ptr += plane_offset;
            top_ptr += top_plane_offset;
        }
    }
}

std::vector<unsigned int> OpenPosePostProcess::getPosePairs(bm::PoseKeyPoints::EModelType model_type) {
    switch (model_type) {
        case bm::PoseKeyPoints::EModelType::BODY_25:
            return {
                1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15,
                17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24
            };
        case bm::PoseKeyPoints::EModelType::COCO_18:
            return {
                1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0,
                15, 15, 17, 2, 16, 5, 17
            };
        default:
            // COCO_18
            return {
                1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0,
                15, 15, 17, 2, 16, 5, 17
            };
    }
}

std::vector<unsigned int> OpenPosePostProcess::getPoseMapIdx(bm::PoseKeyPoints::EModelType model_type) {
    switch (model_type) {
        case bm::PoseKeyPoints::EModelType::BODY_25:
            return {
                26,27, 40,41, 48,49, 42,43, 44,45, 50,51, 52,53, 32,33, 28,29, 30,31, 34,35, 36,37, 38,39, 56,57, 58,59, 62,63, 60,61, 64,65, 46,47, 54,55, 66,67, 68,69, 70,71, 72,73, 74,75, 76,77
            };
        case bm::PoseKeyPoints::EModelType::COCO_18:
            return {
                31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46
            };
        default:
            // COCO_18
            return {
                    31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46
            };
    }
}

int OpenPosePostProcess::getNumberBodyParts(bm::PoseKeyPoints::EModelType model_type) {
    switch (model_type) {
        case bm::PoseKeyPoints::EModelType::BODY_25:
            return 25;
        case bm::PoseKeyPoints::EModelType::COCO_18:
            return 18;
        default:
            // COCO_18
            return 18;
    }
}

void OpenPosePostProcess::connectBodyPartsCpu(std::vector<float>& poseKeypoints, const float* const heatMapPtr,
        const float* const peaksPtr, const cv::Size& heatMapSize, const int maxPeaks,
        const int interMinAboveThreshold, const float interThreshold, const int minSubsetCnt,
        const float minSubsetScore, const float scaleFactor, std::vector<int>& keypointShape, bm::PoseKeyPoints::EModelType modelType)
{
    keypointShape.resize(3);
//    const std::vector<unsigned int> POSE_COCO_PAIRS{ 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17 };
//    const std::vector<unsigned int> POSE_COCO_MAP_IDX{ 31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46 };
    const auto bodyPartPairs = getPosePairs(modelType);
    const auto mapIdx = getPoseMapIdx(modelType);
    const auto numberBodyParts = getNumberBodyParts(modelType); //COCO 18 points

    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

    std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
    const auto subsetCounterIndex = numberBodyParts;
    const auto subsetSize = numberBodyParts + 1;

    const auto peaksOffset = 3 * (maxPeaks + 1);
    const auto heatMapOffset = heatMapSize.area();

    for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
    {
        const auto bodyPartA = bodyPartPairs[2 * pairIndex];
        const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
        const auto* candidateA = peaksPtr + bodyPartA*peaksOffset;
        const auto* candidateB = peaksPtr + bodyPartB*peaksOffset;
        const auto nA = intRound(candidateA[0]);
        const auto nB = intRound(candidateB[0]);

        // add parts into the subset in special case
        if (nA == 0 || nB == 0)
        {
            // Change w.r.t. other
            if (nA == 0) // nB == 0 or not
            {
                for (auto i = 1; i <= nB; i++)
                {
                    bool num = false;
                    const auto indexB = bodyPartB;
                    for (auto j = 0u; j < subset.size(); j++)
                    {
                        const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
                        if (subset[j].first[indexB] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        rowVector[bodyPartB] = bodyPartB*peaksOffset + i * 3 + 2; //store the index
                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
                        const auto subsetScore = candidateB[i * 3 + 2]; //second last number in each row is the total score
                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
            else // if (nA != 0 && nB == 0)
            {
                for (auto i = 1; i <= nA; i++)
                {
                    bool num = false;
                    const auto indexA = bodyPartA;
                    for (auto j = 0u; j < subset.size(); j++)
                    {
                        const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
                        if (subset[j].first[indexA] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        rowVector[bodyPartA] = bodyPartA*peaksOffset + i * 3 + 2; //store the index
                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
                        const auto subsetScore = candidateA[i * 3 + 2]; //second last number in each row is the total score
                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
        }
        else // if (nA != 0 && nB != 0)
        {
            std::vector<std::tuple<double, int, int>> temp;
            const auto numInter = 10;
            const auto* const mapX = heatMapPtr + mapIdx[2 * pairIndex] * heatMapOffset;
            const auto* const mapY = heatMapPtr + mapIdx[2 * pairIndex + 1] * heatMapOffset;
            for (auto i = 1; i <= nA; i++)
            {
                for (auto j = 1; j <= nB; j++)
                {
                    const auto dX = candidateB[j * 3] - candidateA[i * 3];
                    const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
                    const auto normVec = float(std::sqrt(dX*dX + dY*dY));
                    // If the peaksPtr are coincident. Don't connect them.
                    if (normVec > 1e-6)
                    {
                        const auto sX = candidateA[i * 3];
                        const auto sY = candidateA[i * 3 + 1];
                        const auto vecX = dX / normVec;
                        const auto vecY = dY / normVec;

                        auto sum = 0.;
                        auto count = 0;
                        for (auto lm = 0; lm < numInter; lm++)
                        {
                            const auto mX = fastMin(heatMapSize.width - 1, intRound(sX + lm*dX / numInter));
                            const auto mY = fastMin(heatMapSize.height - 1, intRound(sY + lm*dY / numInter));
                            //checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
                            //checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
                            const auto idx = mY * heatMapSize.width + mX;
                            const auto score = (vecX*mapX[idx] + vecY*mapY[idx]);
                            if (score > interThreshold)
                            {
                                sum += score;
                                count++;
                            }
                        }

                        // parts score + connection score
                        if (count > interMinAboveThreshold)
                            temp.emplace_back(std::make_tuple(sum / count, i, j));
                    }
                }
            }

            // select the top minAB connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (!temp.empty())
                std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

            std::vector<std::tuple<int, int, double>> connectionK;

            const auto minAB = fastMin(nA, nB);
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);
            auto counter = 0;
            for (auto row = 0u; row < temp.size(); row++)
            {
                const auto score = std::get<0>(temp[row]);
                const auto x = std::get<1>(temp[row]);
                const auto y = std::get<2>(temp[row]);
                if (!occurA[x - 1] && !occurB[y - 1])
                {
                    connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
                                                             bodyPartB*peaksOffset + y * 3 + 2,
                                                             score));
                    counter++;
                    if (counter == minAB)
                        break;
                    occurA[x - 1] = 1;
                    occurB[y - 1] = 1;
                }
            }

            // Cluster all the body part candidates into subset based on the part connection
            // initialize first body part connection 15&16
            if (pairIndex == 0)
            {
                for (const auto connectionKI : connectionK)
                {
                    std::vector<int> rowVector(numberBodyParts + 3, 0);
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    const auto score = std::get<2>(connectionKI);
                    rowVector[bodyPartPairs[0]] = indexA;
                    rowVector[bodyPartPairs[1]] = indexB;
                    rowVector[subsetCounterIndex] = 2;
                    // add the score of parts and the connection
                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                }
            }
                // Add ears connections (in case person is looking to opposite direction to camera)
            else if (
                    (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                    || ((numberBodyParts == 19 || (numberBodyParts == 25)
                         || numberBodyParts == 59 || numberBodyParts == 65)
                        && (pairIndex==18 || pairIndex==19))
                    )
            {
                for (const auto& connectionKI : connectionK)
                {
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    for (auto& subsetJ : subset)
                    {
                        auto& subsetJFirst = subsetJ.first[bodyPartA];
                        auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
                        if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
                            subsetJFirstPlus1 = indexB;
                        else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
                            subsetJFirst = indexA;
                    }
                }
            }
            else
            {
                if (!connectionK.empty())
                {
                    // A is already in the subset, find its connection B
                    for (auto i = 0u; i < connectionK.size(); i++)
                    {
                        const auto indexA = std::get<0>(connectionK[i]);
                        const auto indexB = std::get<1>(connectionK[i]);
                        const auto score = std::get<2>(connectionK[i]);
                        auto num = 0;
                        for (auto j = 0u; j < subset.size(); j++)
                        {
                            if (subset[j].first[bodyPartA] == indexA)
                            {
                                subset[j].first[bodyPartB] = indexB;
                                num++;
                                subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
                                subset[j].second = subset[j].second + peaksPtr[indexB] + score;
                            }
                        }
                        // if can not find partA in the subset, create a new subset
                        if (num == 0)
                        {
                            std::vector<int> rowVector(subsetSize, 0);
                            rowVector[bodyPartA] = indexA;
                            rowVector[bodyPartB] = indexB;
                            rowVector[subsetCounterIndex] = 2;
                            const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                            subset.emplace_back(std::make_pair(rowVector, subsetScore));
                        }
                    }
                }
            }
        }
    }

    // Delete people below the following thresholds:
    // a) minSubsetCnt: removed if less than minSubsetCnt body parts
    // b) minSubsetScore: removed if global score smaller than this
    // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
    auto numberPeople = 0;
    std::vector<int> validSubsetIndexes;
    validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
    for (auto index = 0u; index < subset.size(); index++)
    {
        const auto subsetCounter = subset[index].first[subsetCounterIndex];
        const auto subsetScore = subset[index].second;
        if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
        {
            numberPeople++;
            validSubsetIndexes.emplace_back(index);
            if (numberPeople == POSE_MAX_PEOPLE)
                break;
        }
        else if (subsetCounter < 1)
            printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
    }

    // Fill and return poseKeypoints
    keypointShape = { numberPeople, (int)numberBodyParts, 3 };
    if (numberPeople > 0)
        poseKeypoints.resize(numberPeople * (int)numberBodyParts * 3);
    else
        poseKeypoints.clear();

    for (auto person = 0u; person < validSubsetIndexes.size(); person++)
    {
        const auto& subsetI = subset[validSubsetIndexes[person]].first;
        for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
        {
            const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
            const auto bodyPartIndex = subsetI[bodyPart];
            if (bodyPartIndex > 0)
            {
                poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
                poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
                poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
            }
            else
            {
                poseKeypoints[baseOffset] = 0.f;
                poseKeypoints[baseOffset + 1] = 0.f;
                poseKeypoints[baseOffset + 2] = 0.f;
            }
        }
    }
}

void OpenPosePostProcess::renderKeypointsCpu(cv::Mat& frame, const std::vector<float>& keypoints, std::vector<int> keyshape, const std::vector<unsigned int>& pairs,
                        const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                        const float threshold, float scale)
{
    // Get frame channels
    const auto width = frame.cols;
    const auto height = frame.rows;
    const auto area = width * height;

    // Parameters
    const auto lineType = 8;
    const auto shift = 0;
    const auto numberColors = colors.size();
    const auto thresholdRectangle = 0.1f;
    const auto numberKeypoints = keyshape[1];

    // Keypoints
    for (auto person = 0; person < keyshape[0]; person++)
    {
        {
            const auto ratioAreas = 1;
            // Size-dependent variables
            const auto thicknessRatio = fastMax(intRound(std::sqrt(area)*thicknessCircleRatio * ratioAreas), 1);
            // Negative thickness in cv::circle means that a filled circle is to be drawn.
            const auto thicknessCircle = (ratioAreas > 0.05 ? thicknessRatio : -1);
            const auto thicknessLine = 2;// intRound(thicknessRatio * thicknessLineRatioWRTCircle);
            const auto radius = thicknessRatio / 2;

            // Draw lines
            for (auto pair = 0u; pair < pairs.size(); pair += 2)
            {
                const auto index1 = (person * numberKeypoints + pairs[pair]) * keyshape[2];
                const auto index2 = (person * numberKeypoints + pairs[pair + 1]) * keyshape[2];
                if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
                {
                    const auto colorIndex = pairs[pair + 1] * 3; // Before: colorIndex = pair/2*3;
                    const cv::Scalar color{ colors[(colorIndex+2) % numberColors],
                                            colors[(colorIndex + 1) % numberColors],
                                            colors[(colorIndex + 0) % numberColors] };
                    const cv::Point keypoint1{ intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale) };
                    const cv::Point keypoint2{ intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale) };
                    cv::line(frame, keypoint1, keypoint2, color, thicknessLine, lineType, shift);
                }
            }

            // Draw circles
            for (auto part = 0; part < numberKeypoints; part++)
            {
                const auto faceIndex = (person * numberKeypoints + part) * keyshape[2];
                if (keypoints[faceIndex + 2] > threshold)
                {
                    const auto colorIndex = part * 3;
                    const cv::Scalar color{ colors[(colorIndex+2) % numberColors],
                                            colors[(colorIndex + 1) % numberColors],
                                            colors[(colorIndex + 0) % numberColors] };
                    const cv::Point center{ intRound(keypoints[faceIndex] * scale), intRound(keypoints[faceIndex + 1] * scale) };
                    cv::circle(frame, center, radius, color, thicknessCircle, lineType, shift);
                }
            }
        }
    }
}

void OpenPosePostProcess::renderPoseKeypointsCpu(cv::Mat& frame, const std::vector<float>& poseKeypoints, std::vector<int> keyshape,
                            const float renderThreshold, float scale, const bool blendOriginalFrame)
{
    // Background
    if (!blendOriginalFrame)
        frame.setTo(0.f); // [0-255]

    // Parameters
    const auto thicknessCircleRatio = 1.f / 75.f;
    const auto thicknessLineRatioWRTCircle = 0.75f;
    const auto& pairs = POSE_COCO_PAIRS_RENDER;

    // Render keypoints
    renderKeypointsCpu(frame, poseKeypoints, keyshape, pairs, POSE_COCO_COLORS_RENDER, thicknessCircleRatio,
                       thicknessLineRatioWRTCircle, renderThreshold, scale);
}

int OpenPosePostProcess::getKeyPoints(bm::BMNNTensorPtr outputTensorPtr, cv::Size& netInputSize, cv::Size& originSize,
        std::vector<bm::PoseKeyPoints> &body_keypoints, bm::PoseKeyPoints::EModelType model_type) {
    OpenPosePostProcess postProcess;
    int n = outputTensorPtr->get_num();
    int chan_num = outputTensorPtr->get_shape()->dims[1];
    int net_output_height = outputTensorPtr->get_shape()->dims[2];
    int net_output_width = outputTensorPtr->get_shape()->dims[3];
    PoseBlobPtr resizedBlob = std::make_shared<PoseBlob>(1, chan_num, netInputSize.height, netInputSize.width);
    for(int batch_idx = 0;batch_idx < n; ++batch_idx) {
        int batch_byte_size = chan_num*net_output_height*net_output_width;
        float *base = outputTensorPtr->get_cpu_data() + batch_byte_size * batch_idx;
        for (int ch = 0; ch < chan_num; ++ch) {
            int ch_area = net_output_height* net_output_width;
            cv::Mat src(net_output_height, net_output_width, CV_32F, base +  ch_area * ch);
            cv::Mat dst(resizedBlob->height(), resizedBlob->width(), CV_32F,
                    resizedBlob->data() + netInputSize.height*netInputSize.width * ch);
            cv::resize(src, dst, netInputSize, 0, 0, cv::INTER_CUBIC);
        }
        PoseBlobPtr nms_blob;
        if (model_type == bm::PoseKeyPoints::EModelType::COCO_18) {
            nms_blob = std::make_shared<PoseBlob>(1, 56, POSE_MAX_PEOPLE + 1, 3);
        } else {
            nms_blob = std::make_shared<PoseBlob>(1, 77, POSE_MAX_PEOPLE + 1, 3);
        }

        postProcess.Nms(resizedBlob, nms_blob, 0.05);

        bm::PoseKeyPoints poseKeyPoints;
        poseKeyPoints.width = netInputSize.width;
        poseKeyPoints.height = netInputSize.height;
        connectBodyPartsCpu(poseKeyPoints.keypoints, resizedBlob->data(), nms_blob->data(), netInputSize,
                            POSE_MAX_PEOPLE, 9, 0.05, 3, 0.4, 1, poseKeyPoints.shape,
                            model_type);

        //cv::Mat img = cv::Mat::zeros(netInputSize, CV_8UC3);
        //renderPoseKeypointsCpu(img, keypoints, shape, 0.05, 1.0);
        poseKeyPoints.modeltype = model_type;
        body_keypoints.push_back(poseKeyPoints);
    }
}

