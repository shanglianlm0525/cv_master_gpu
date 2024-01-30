

## CV_MASTER_CPU使用



## 1 概述

基于openvino和yolov8做的支持图像分类，目标检测，实例分割的算法库，支持CPU运行。



## 2 环境配置

属性 --> VC++ 目录 --> 包含目录

```bash
C:\Intel\openvino_2023.0.0\runtime\include
E:\opencv470\build\include
E:\cpp_code\cv_master_cpu
```

属性 --> VC++ 目录 --> 库目录

```bash
C:\Intel\openvino_2023.0.0\runtime\lib\intel64\Release
E:\opencv470\build\x64\vc16\lib
```

属性 --> 链接器 --> 输入 --> 附加依赖项

```bash
opencv_world470.lib
openvino.lib
```

将 opencv_world455.dll，paddle_inference.dll，paddle2onnx.dll 和 onnxruntime.dll复制到项目的x64\Release目录下

动态链接库 配置
将 C:\Intel\openvino_2023.0.0\runtime\bin\intel64\Release目录下的

```bash
openvino.dll
openvino_intel_cpu_plugin.dll
openvino_ir_frontend.dll
```

将 C:\Intel\openvino_2023.0.0\runtime\3rdparty\tbb\bin目录下的

```bash
tbb12.dll
```


将 E:\opencv470\mybuild\x64\vc16\bin 目录下的

```bash
opencv_world470.dll
```

移动到 可执行文件目录 或者将三个路径加入系统目录。

## 3 调用接口：

```c++
struct CvMasterCpuDll
{
	virtual void Release() = 0;

	// preprocess det
	virtual cv::Mat preprocess_det(cv::Mat img, std::vector<float>& scales, cv::Mat& resize_img) = 0;

	// preprocess seg
	virtual cv::Mat preprocess_seg(cv::Mat img, std::vector<float>& scales, cv::Mat& resize_img) = 0;

	// preprocess cls
	virtual cv::Mat preprocess_cls(cv::Mat img, cv::Mat& resize_img) = 0;

	// model det
	virtual int infer_det(cv::Mat img, std::vector<float> scales, ModelPara mPara, DetectInfo* dInfoList, int* dNum) = 0;

	// model seg
	virtual int infer_seg(cv::Mat img, std::vector<float> scales, ModelPara mPara, DetectInfo* dInfoList, int* dNum) = 0;

	// model cls
	virtual int infer_cls(std::vector<cv::Mat> imgs, DetectInfo* imgsInfoList, ModelPara mPara, DetectInfo* dInfoList, int* dNum) = 0;
};



extern "C" CVMASTERCPUDLL_API int  verification_authorization_code(const char* license_path, const char* pj_name);


extern "C" CVMASTERCPUDLL_API CvMasterCpuDll * __stdcall GetObj(const char* license_path, const char* pj_name, const char* model_path, std::vector<uint8_t>&model_data, const char* weight_path, std::vector<uint8_t>&weight_data, int max_batch_size = 1, int async_num = 0, int type = 0);
```

 