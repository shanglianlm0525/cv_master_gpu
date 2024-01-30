#pragma once
#include "include/utils.h"
#include "include/kernel_function.h"


void decodeDevice(InPara param, float* src, int srcWidth, int srcHeight, int srcLength, float* dst, int dstWidth, int dstHeight, int nType=0);
void transposeDevice(InPara param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight);
