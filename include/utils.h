#pragma once
#include "common_include.h"

enum class InputStream { IMAGE, VIDEO, CAMERA };

enum class ColorMode { RGB, GRAY };

struct AffineMat
{
    float v0, v1, v2;
    float v3, v4, v5;
};

struct InPara
{
    int num_class{ 80 }; // coco 
    std::vector<std::string> input_names = { "images" };
    std::vector<std::string> output_names = { "output0" };


    bool fixed_shape{ true };
    int batch_size = 1;
    int nChannel = 3;
    int nOrgHeight = 0, nOrgWidth = 0;
    int nHeight = 0, nWidth = 0;

    float scale{ 255.f };
    float means[3] = { 0.f, 0.f, 0.f };
    float stds[3] = { 1.f, 1.f, 1.f };

    float iou_thresh = 0.45f;
    float conf_thresh = 0.01f;

    int topK{ 1000 };
    int nMaxObjNum = 1000;
};

/*
struct Box
{
    float left, top, right, bottom, confidence;
    int label;
    std::vector<cv::Point2i> land_marks;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label) :
        left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}

    Box(float left, float top, float right, float bottom, float confidence, int label, int numLandMarks) :
        left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label)
    {
        land_marks.reserve(numLandMarks);
    }
};
*/

const std::vector<cv::Scalar> colors{
            cv::Scalar(128, 77, 207),cv::Scalar(65, 32, 208),cv::Scalar(0, 224, 45),cv::Scalar(3, 141, 219),cv::Scalar(80, 239, 253),cv::Scalar(239, 184, 12),
            cv::Scalar(7, 144, 145),cv::Scalar(161, 88, 57),cv::Scalar(0, 166, 46),cv::Scalar(218, 113, 53),cv::Scalar(193, 33, 128),cv::Scalar(190, 94, 113),
            cv::Scalar(113, 123, 232),cv::Scalar(69, 205, 80),cv::Scalar(18, 170, 49),cv::Scalar(89, 51, 241),cv::Scalar(153, 191, 154),cv::Scalar(27, 26, 69),
            cv::Scalar(20, 186, 194),cv::Scalar(210, 202, 167),cv::Scalar(196, 113, 204),cv::Scalar(9, 81, 88),cv::Scalar(191, 162, 67),cv::Scalar(227, 73, 120),
            cv::Scalar(177, 31, 19),cv::Scalar(133, 102, 137),cv::Scalar(146, 72, 97),cv::Scalar(145, 243, 208),cv::Scalar(2, 184, 176),cv::Scalar(219, 220, 93),
            cv::Scalar(238, 253, 234),cv::Scalar(197, 169, 160),cv::Scalar(204, 201, 106),cv::Scalar(13, 24, 129),cv::Scalar(40, 38, 4),cv::Scalar(5, 41, 34),
            cv::Scalar(46, 94, 129),cv::Scalar(102, 65, 107),cv::Scalar(27, 11, 208),cv::Scalar(191, 240, 183),cv::Scalar(225, 76, 38),cv::Scalar(193, 89, 124),
            cv::Scalar(30, 14, 175),cv::Scalar(144, 96, 90),cv::Scalar(181, 186, 86),cv::Scalar(102, 136, 34),cv::Scalar(158, 71, 15),cv::Scalar(183, 81, 247),
            cv::Scalar(73, 69, 89),cv::Scalar(123, 73, 232),cv::Scalar(4, 175, 57),cv::Scalar(87, 108, 23),cv::Scalar(105, 204, 142),cv::Scalar(63, 115, 53),
            cv::Scalar(105, 153, 126),cv::Scalar(247, 224, 137),cv::Scalar(136, 21, 188),cv::Scalar(122, 129, 78),cv::Scalar(145, 80, 81),cv::Scalar(51, 167, 149),
            cv::Scalar(162, 173, 20),cv::Scalar(252, 202, 17),cv::Scalar(10, 40, 3),cv::Scalar(150, 90, 254),cv::Scalar(169, 21, 68),cv::Scalar(157, 148, 180),
            cv::Scalar(131, 254, 90),cv::Scalar(7, 221, 102),cv::Scalar(19, 191, 184),cv::Scalar(98, 126, 199),cv::Scalar(210, 61, 56),cv::Scalar(252, 86, 59),
            cv::Scalar(102, 195, 55),cv::Scalar(160, 26, 91),cv::Scalar(60, 94, 66),cv::Scalar(204, 169, 193),cv::Scalar(126, 4, 181),cv::Scalar(229, 209, 196),
            cv::Scalar(195, 170, 186),cv::Scalar(155, 207, 148)
};

void saveBinaryFile(float* vec, size_t len, const std::string& file);

std::vector<uint8_t> readBinaryFile(const std::string& file);

std::vector<unsigned char> loadModel(const std::string& file);



void computeAffineMat(InPara m_param, AffineMat& m_dst2src);


class DeviceTimer
{
public:
    DeviceTimer();
    float getUsedTime();
    // overload
    DeviceTimer(cudaStream_t ctream);
    float getUsedTime(cudaStream_t ctream);

    ~DeviceTimer();

private:
    cudaEvent_t start, end;
};