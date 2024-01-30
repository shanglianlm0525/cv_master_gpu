#pragma once


#define CVMASTERGPUALG_EXPORTS

#define WIN32_LEAN_AND_MEAN             // 从 Windows 头文件中排除极少使用的内容
// Windows 头文件
#include <windows.h>

#ifdef CVMASTERGPUALG_EXPORTS
#define CVMASTERGPUDLL_API __declspec(dllexport)
#else
#define CVMASTERGPUDLL_API __declspec(dllimport)
#endif


#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/augments.h"



struct CvMasterGpuDll
{
	virtual void Release() = 0;

	// preprocess det
	virtual void preprocess_det(std::vector<cv::Mat> imgsBatch) = 0;

	// preprocess cls
	virtual void preprocess_cls(std::vector<cv::Mat> imgsBatch) = 0;

	// preprocess seg
	virtual void preprocess_seg(std::vector<cv::Mat> imgsBatch) = 0;

	// model det
	virtual int infer_det(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum) = 0;

	// model cls
	virtual int infer_cls(std::vector<cv::Mat> imgsBatch, DetectInfo* imgsInfoList, ModelPara mPara, DetectInfo* dInfoList, int* dNum) = 0;

	// model seg
	virtual int infer_seg(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum) = 0;
};



extern "C" CVMASTERGPUDLL_API int  verification_authorization_code(const char* license_path, const char* pj_name);


extern "C" CVMASTERGPUDLL_API CvMasterGpuDll * __stdcall GetObj(const char* license_path, const char* pj_name, InPara param, const char* model_path, std::vector<unsigned char> &model_data, int type);
