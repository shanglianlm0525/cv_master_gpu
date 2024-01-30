#pragma once
#include "include/utils.h"
#include "include/common.h"
#include "include/augments.h"

class CvGpuAlgCls
{
public:
    CvGpuAlgCls() {};
    // CvGpuAlgCls(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data);
    ~CvGpuAlgCls();

    virtual void init(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data);
    virtual void preprocess(std::vector<cv::Mat> imgs);
    virtual int infer(std::vector<cv::Mat> imgs, DetectInfo* imgsInfoList, ModelPara mPara, DetectInfo* dInfoList, int* dNum);

private:
    virtual void postprocess(std::vector<cv::Mat> imgsBatch);



public:
    int i = 0, j = 0;


protected:
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
    InPara m_param;
    nvinfer1::Dims m_input_dims, m_output_dims;
    int m_output_area;
    int m_total_objects;
    AffineMat m_dst2src;

    // input
    unsigned char* m_input_src_device;
    float* m_input_resize_device;

    float* m_input_rgb_device;
    float* m_input_norm_device;
    float* m_input_hwc_device;

    // output
    float* m_output_src_device;
    float* m_output_objects_host;

    /*
    float* m_output_objects_device;
    float* m_output_objects_host;
    int m_output_objects_width;
    int* m_output_idx_device;
    float* m_output_conf_device;
    
    float* m_output_src_transpose_device;
    */
};
