// cv_master_gpu.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include "include/utils.h"
#include "cv_master_gpu.h"
#include "include/cv_gpu_alg_det.h"
#include "include/cv_gpu_alg_seg.h"
#include "include/cv_gpu_alg_cls.h"

std::string setValidTime;
int nImgCount = 0;
int initVerifyFlag = 0;


typedef const char* (*obtainAuthorizationCode)();

const char* obtainDllValue(std::string auth_path) {
	HMODULE hDll; // DLL的句柄
	obtainAuthorizationCode myFunc; // 函数指针

	std::wstring wideDir(auth_path.begin(), auth_path.end());

	// 通过加载库来获取DLL的句柄
	hDll = LoadLibrary(wideDir.c_str()); // 假设原来的DLL名称为old_dll_name.dll
	if (hDll == NULL) {
		std::cout << "Failed to load DLL!" << std::endl;
		return "";
	}

	// 获取DLL中的函数地址
	myFunc = (obtainAuthorizationCode)GetProcAddress(hDll, "obtainAuthorizationCode"); // 假设DLL中的函数名为MyFunction
	if (myFunc == NULL) {
		std::cout << "Failed to get function address!" << std::endl;
		return "";
	}

	// 调用DLL中的函数
	const char* result = myFunc();
	std::string retStr(result);
	// std::cout << result << std::endl; // 输出读取的字符串

	// 卸载DLL
	FreeLibrary(hDll);

	return retStr.c_str();
}


CVMASTERGPUDLL_API int verification_authorization_code(const char* license_path, const char* pj_name) {


	const char* readStr = obtainDllValue(license_path);
	// std::cout << "Read String: " << readStr << std::endl; // 输出读取的字符串

	std::string decry_str = DecryptionAES(readStr);
	// std::cout << "decry_str: " << decry_str << std::endl; // 输出读取的字符串


	char delimiter = '#';
	std::vector<std::string> splitStr = split_str(decry_str, delimiter);

	std::string decry_project_id = splitStr[0]; // reserved for future
	std::string decry_pj_name = splitStr[1];
	setValidTime = splitStr[2];

	std::string pj_name_str(pj_name);
	pj_name_str = pj_name_str.substr(0, pj_name_str.length() - 4);

	// std::cout << decry_project_id << " " << decry_pj_name << " " << setValidTime << std::endl;
	// std::cout << pj_name_str << " " << decry_pj_name << " " << (pj_name_str != decry_pj_name) << std::endl;
	// std::cout << verification_localTime(setValidTime)  << std::endl;

	if (pj_name_str != decry_pj_name or verification_localTime(setValidTime)) {
		return 1;
	}

	return 0;
}


int verify_local_time() {
	if (nImgCount > 2592000) {
		nImgCount = 0;

		if (setValidTime.empty() or verification_localTime(setValidTime)) {
			return -2;
		}
	}
	nImgCount = nImgCount + 1;

	return 0;
}



class CvMasterGpu : public CvMasterGpuDll
{
public:

	CvGpuAlgDet detect_model;
	CvGpuAlgSeg segment_model;
	CvGpuAlgCls classify_model;

	int resize_width, resize_height;
	cv::Scalar border = cv::Scalar(114, 114, 114);
	cv::Scalar border1d = cv::Scalar(114);

	CvMasterGpu(const char* license_path, const char* pj_name, InPara param, const char* model_path, std::vector<unsigned char> &model_data, int type = 0);


	// preprocess det
	void preprocess_det(std::vector<cv::Mat> imgsBatch);

	// preprocess seg
	void preprocess_seg(std::vector<cv::Mat> imgsBatch);

	// preprocess cls
	void preprocess_cls(std::vector<cv::Mat> imgs);

	// model det
	int infer_det(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum);

	// model seg
	int infer_seg(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum);

	// model cls
	int infer_cls(std::vector<cv::Mat> imgs, DetectInfo* imgsInfoList, ModelPara mPara, DetectInfo* dInfoList, int* dNum);

	void Release();
};



CvMasterGpu::CvMasterGpu(const char* license_path, const char* pj_name, InPara param, const char* model_path, std::vector<unsigned char> &model_data, int type)
{                                                                                          
	if (initVerifyFlag == 0) {
		if (verification_authorization_code(license_path, pj_name)) {
			return;
		}
		initVerifyFlag = 1;
	}


	if (type == 1) {
		classify_model.init(param, model_path, model_data);
	}
	else if (type == 2) {
		segment_model.init(param, model_path, model_data);
	}
	else {
		detect_model.init(param, model_path, model_data);
	}
}


void CvMasterGpu::Release()
{
	delete this;
}




void CvMasterGpu::preprocess_seg(std::vector<cv::Mat> imgsBatch)
{
	segment_model.preprocess(imgsBatch);
}

void CvMasterGpu::preprocess_det(std::vector<cv::Mat> imgsBatch)
{
	detect_model.preprocess(imgsBatch);
}

void CvMasterGpu::preprocess_cls(std::vector<cv::Mat> imgsBatch)
{

	classify_model.preprocess(imgsBatch);
}


int CvMasterGpu::infer_det(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum)
{
	if (verify_local_time()) {
		return -2;
	}
	return detect_model.infer(imgsBatch, mPara, dInfoList, dNum);
}


int CvMasterGpu::infer_seg(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum)
{
	if (verify_local_time()) {
		return -2;
	}
	return segment_model.infer(imgsBatch, mPara, dInfoList, dNum);
}


int CvMasterGpu::infer_cls(std::vector<cv::Mat> imgsBatch, DetectInfo* imgsInfoList, ModelPara mPara, DetectInfo* dInfoList, int* dNum)
{
	if (verify_local_time()) {
		return -2;
	}
	return classify_model.infer(imgsBatch, imgsInfoList, mPara, dInfoList, dNum);
}



CVMASTERGPUDLL_API CvMasterGpuDll* __stdcall GetObj(const char* license_path, const char* pj_name, InPara param, const char* model_path, std::vector<unsigned char>& model_data, int type)
{
	return new CvMasterGpu(license_path, pj_name, param, model_path, model_data, type);
}



void setParameters_det(InPara& initParameter)
{
	initParameter.input_names = { "images" };
	initParameter.output_names = { "output0" };
	initParameter.conf_thresh = 0.25f;
	initParameter.iou_thresh = 0.45f;

	initParameter.fixed_shape = false;
	initParameter.nOrgHeight = -1;
	initParameter.nOrgWidth = -1;
}



void detection_demo() {

	const char* license_path = "E:\\cpp_code\\generate_project_code\\x64\\Release\\w5nGcPfKD5aCj8HV1rV82Q==1.dll";
	const char* pj_name = "BotProject.dll";

	// parameters
	InPara param;

	/*
	const char* model_path = "E:\\cpp_code\\weights\\trt_weights\\yolov8s.trt";
	std::string image_path = "E:\\cpp_code\\images\\bus.jpg";
	cv::Mat frame = cv::imread(image_path);
	setParameters(param);
	*/

	const char* model_path = "E:\\cpp_code\\weights\\trt_weights\\stringing_glue_jd.trt";
	std::string image_path = "E:\\cpp_code\\images\\20231026164442A.bmp";
	cv::Mat frame = cv::imread(image_path, 0);
	setParameters_det(param);

	param.nOrgHeight = frame.rows;
	param.nOrgWidth = frame.cols;

	std::vector<unsigned char> model_data;


	CvMasterGpuDll* base_model = GetObj(license_path, pj_name, param, model_path, model_data, 0);



	ModelPara modelPara;
	modelPara.score_threshold = 0.25f;     //  score threshold
	modelPara.nms_threshold = 0.5f;       // threshold for NMS
	modelPara.nMaxObjNum = 1000;
	modelPara.flag = 1;

	DetectInfo imgsInfoList[1000];

	const int nCount = 1000;
	DetectInfo dDefectList[nCount];

	int object_num;

	std::vector<cv::Mat> imgs_batch;
	imgs_batch.emplace_back(frame.clone());


	DeviceTimer d_t3; base_model->preprocess_det(imgs_batch);	float t3 = d_t3.getUsedTime();
	DeviceTimer d_t4; base_model->infer_det(imgs_batch, modelPara, dDefectList, &object_num);	float t4 = d_t4.getUsedTime();
	sample::gLogInfo << "preprocess time = " << t3 / param.batch_size << std::endl;
	sample::gLogInfo << "infer time = " << t4 / param.batch_size << std::endl;

	sample::gLogInfo << "infer time = " << object_num << std::endl;

	// show
	for (size_t bi = 0; bi < imgs_batch.size(); bi++)
	{
		cv::Mat img_show = imgs_batch[bi];
		if (img_show.channels() == 1) {
			cv::cvtColor(img_show, img_show, cv::COLOR_GRAY2BGR);
		}
		for (int i = 0; i < object_num; ++i)
		{
			std::cout << dDefectList[i].x << " " << dDefectList[i].y << " " << dDefectList[i].width << " " << dDefectList[i].height << std::endl;
			int offset = 5;
			cv::rectangle(img_show, cv::Point(dDefectList[i].x - offset, dDefectList[i].y - offset), 
				cv::Point(dDefectList[i].x + dDefectList[i].width + offset * 2, dDefectList[i].y + dDefectList[i].height + offset * 2), colors[dDefectList[i].det_id], 2, 8, 0);//黄色矩形框
			// cv::putText(img_show, std::to_string(dDefectList[i].det_id) +"_"+ std::to_string(dDefectList[i].det_score), cv::Point(dDefectList[i].x, dDefectList[i].y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 4);
		}
		cv::resize(img_show, img_show, cv::Size(1600, 1200));
		cv::imshow("img_show", img_show);
	}

	imgs_batch.clear();

	cv::waitKey(0);

	base_model->Release();
}



void setParameters_cls(InPara& initParameter)
{
	initParameter.input_names = { "images" };
	initParameter.output_names = { "output0" };
	initParameter.conf_thresh = 0.25f;
	initParameter.iou_thresh = 0.45f;

	initParameter.fixed_shape = true;
	initParameter.nOrgHeight = -1;
	initParameter.nOrgWidth = -1;
}



void classification_demo() {

	const char* license_path = "E:\\cpp_code\\generate_project_code\\x64\\Release\\w5nGcPfKD5aCj8HV1rV82Q==1.dll";
	const char* pj_name = "BotProject.dll";

	// parameters
	InPara param;

	const char* model_path = "E:\\cpp_code\\weights\\trt_weights\\stringing_glue_jd_cls.trt";
	std::string image_path = "E:\\cpp_code\\cv_master_gpu\\crop_img1.bmp";
	cv::Mat frame = cv::imread(image_path, 0);

	std::string image_path2 = "E:\\cpp_code\\cv_master_gpu\\crop_img2.bmp";
	cv::Mat frame2 = cv::imread(image_path2, 0);
	setParameters_cls(param);

	param.nOrgHeight = 32;
	param.nOrgWidth = 32;

	std::vector<unsigned char> model_data;

	CvMasterGpuDll* base_model = GetObj(license_path, pj_name, param, model_path, model_data, 1);

	cv::resize(frame, frame, cv::Size(32, 32));
	cv::resize(frame2, frame2, cv::Size(32, 32));

	std::vector<cv::Mat> imgs;
	for (int i = 0; i < 500; ++i) {
		imgs.emplace_back(frame.clone());
		imgs.emplace_back(frame2.clone());
	}
	

	ModelPara modelPara;
	modelPara.score_threshold = 0.25f;     //  score threshold
	modelPara.nms_threshold = 0.5f;       // threshold for NMS
	modelPara.nMaxObjNum = 1000;
	modelPara.flag = 1;

	DetectInfo imgsInfoList[1000];

	const int nCount = 1000;
	DetectInfo dDefectList[nCount];


	int object_num;

	DeviceTimer d_t4; base_model->infer_cls(imgs, imgsInfoList, modelPara, dDefectList, &object_num);	float t4 = d_t4.getUsedTime();
	sample::gLogInfo << "infer time = " << t4 / param.batch_size << std::endl;



	cv::waitKey(0);

	base_model->Release();
}


void setParameters_seg(InPara& initParameter)
{
	initParameter.input_names = { "images" };
	initParameter.output_names = { "output0" };
	initParameter.conf_thresh = 0.25f;
	initParameter.iou_thresh = 0.45f;

	initParameter.fixed_shape = false;
	initParameter.nOrgHeight = -1;
	initParameter.nOrgWidth = -1;
}



void segmentation_demo() {

	const char* license_path = "E:\\cpp_code\\generate_project_code\\x64\\Release\\w5nGcPfKD5aCj8HV1rV82Q==1.dll";
	const char* pj_name = "BotProject.dll";

	// parameters
	InPara param;

	
	const char* model_path = "E:\\cpp_code\\weights\\trt_weights\\yolov8n-seg.trt";
	std::string image_path = "E:\\cpp_code\\images\\bus.jpg";
	cv::Mat frame = cv::imread(image_path);
	setParameters_seg(param);

	param.nOrgHeight = frame.rows;
	param.nOrgWidth = frame.cols;

	std::vector<unsigned char> model_data;

	CvMasterGpuDll* base_model = GetObj(license_path, pj_name, param, model_path, model_data, 2);


	ModelPara modelPara;
	modelPara.score_threshold = 0.25f;     //  score threshold
	modelPara.nms_threshold = 0.5f;       // threshold for NMS
	modelPara.nMaxObjNum = 1000;
	modelPara.flag = 1;

	DetectInfo imgsInfoList[1000];

	const int nCount = 1000;
	DetectInfo dDefectList[nCount];

	int object_num;

	std::vector<cv::Mat> imgs_batch;
	imgs_batch.emplace_back(frame.clone());


	DeviceTimer d_t3; base_model->preprocess_seg(imgs_batch);	float t3 = d_t3.getUsedTime();
	DeviceTimer d_t4; base_model->infer_seg(imgs_batch, modelPara, dDefectList, &object_num);	float t4 = d_t4.getUsedTime();
	sample::gLogInfo << "preprocess time = " << t3 / param.batch_size << std::endl;
	sample::gLogInfo << "infer time = " << t4 / param.batch_size << std::endl;

	sample::gLogInfo << "object_num = " << object_num << std::endl;

	// show
	for (size_t bi = 0; bi < imgs_batch.size(); bi++)
	{
		cv::Mat img_show = imgs_batch[bi];
		if (img_show.channels() == 1) {
			cv::cvtColor(img_show, img_show, cv::COLOR_GRAY2BGR);
		}
		for (int i = 0; i < object_num; ++i)
		{
			std::cout << dDefectList[i].x << " " << dDefectList[i].y << " " << dDefectList[i].width << " " << dDefectList[i].height<<" "<< dDefectList[i].det_score << std::endl;
			int offset = 5;
			cv::rectangle(img_show, cv::Point(dDefectList[i].x - offset, dDefectList[i].y - offset),
				cv::Point(dDefectList[i].x + dDefectList[i].width + offset * 2, dDefectList[i].y + dDefectList[i].height + offset * 2), colors[dDefectList[i].det_id], 2, 8, 0);//黄色矩形框
			// cv::putText(img_show, std::to_string(dDefectList[i].det_id) +"_"+ std::to_string(dDefectList[i].det_score), cv::Point(dDefectList[i].x, dDefectList[i].y), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 4);
		}
		cv::resize(img_show, img_show, cv::Size(1600, 1200));
		cv::imshow("img_show", img_show);
	}

	imgs_batch.clear();

	cv::waitKey(0);

	base_model->Release();
}



int main()
{
	// detection_demo();
	// classification_demo();
	segmentation_demo();

	return 0;
}

