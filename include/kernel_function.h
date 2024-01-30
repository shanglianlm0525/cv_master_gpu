#pragma once

#include "common_include.h"
#include "utils.h"

#define BLOCK_SIZE 8
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)


bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);


//note: resize rgb with padding
void resizeDevice(const int& batch_size, float* src, int src_width, int src_height, int srcChannel,
    float* dst, int dstWidth, int dstHeight,
    float paddingValue, AffineMat matrix);

//overload:resize rgb with padding, but src's type is uin8
void resizeDevice(const int& batch_size, unsigned char* src, int src_width, int src_height, int srcChannel,
    float* dst, int dstWidth, int dstHeight,
    float paddingValue, AffineMat matrix);

// overload: resize rgb/gray without padding
void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight, int srcChannel,
    float* dst, int dstWidth, int dstHeight,
    ColorMode mode, AffineMat matrix);

void bgr2rgbDevice(const int& batch_size, float* src, int srcWidth, int srcHeight, int srcChannel,
    float* dst, int dstWidth, int dstHeight);

void normDevice(const int& batch_size, float* src, int srcWidth, int srcHeight, int srcChannel,
    float* dst, int dstWidth, int dstHeight,
    InPara norm_param);

void hwc2chwDevice(const int& batch_size, float* src, int srcWidth, int srcHeight, int srcChannel,
    float* dst, int dstWidth, int dstHeight);

// add by liumin, 20231229
void softmaxDevice(float* src, int batch_size, int num_class);


// nms fast
void nmsDeviceV1(InPara param, float* src, int srcWidth, int srcHeight, int srcChannel, int srcArea);

// nms sort
void nmsDeviceV2(InPara param, float* src, int srcWidth, int srcHeight, int srcChannel, int srcArea,
    int* idx, float* conf);



