

## CV_MASTER_GPU使用



## 1 概述

基于tensorRT和yolov8做的支持图像分类，目标检测，实例分割的算法库，支持NVIDIA GPU运行。



## 2 环境配置

属性 --> VC++ 目录 --> 包含目录

```bash
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
E:\TensorRT-8.6.0.12\include
E:\opencv470\build\include
E:\cpp_code\cv_master_gpu
```

属性 --> VC++ 目录 --> 库目录

```bash
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
E:\TensorRT-8.6.0.12\lib
E:\opencv470\build\x64\vc16\lib
```

属性 --> 链接器 --> 输入 --> 附加依赖项

```bash
opencv_world470.lib
nvinfer.lib
cudart.lib
nvonnxparser.lib
nvinfer_plugin.lib
nvparsers.lib
```


将 E:\opencv470\mybuild\x64\vc16\bin 目录下的

```bash
opencv_world470.dll
```

移动到 可执行文件目录 或者将路径加入系统目录。

## 3 调用接口：

```c++
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
```

 