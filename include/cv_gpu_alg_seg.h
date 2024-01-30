#pragma once
#include "include/utils.h"
#include "include/augments.h"

class CvGpuAlgSeg
{
public:
    CvGpuAlgSeg() {};
    // CvGpuAlgSeg(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data);
    ~CvGpuAlgSeg();

    virtual void init(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data);
    virtual void preprocess(std::vector<cv::Mat> imgsBatch);
    virtual int infer(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum);

private:
    virtual void postprocess(std::vector<cv::Mat> imgsBatch);


public:
    int i = 0, j = 0;
    float f_scale;
    int resize_width, resize_height;
    int padding_width, padding_height;
    cv::Scalar border = cv::Scalar(114, 114, 114);
    cv::Scalar border1d = cv::Scalar(114);

    DetectInfo dInfo;

protected:
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
    InPara m_param;
    nvinfer1::Dims m_input_dims, m_output_dims, m_output_seg_dims;
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
    float* m_output_objects_device;
    float* m_output_objects_host;
    float* m_output_seg_device; // eg:116 * 8400, 116=4+80+32
    float* m_output_seg_host;
    int m_output_objects_width; // 39 = 32 + 7, 7:left, top, right, bottom, confidence, class, keepflag; 
    int m_output_src_width; // 116 = 4+80+32, 4:xyxy; 80:coco label; 32:seg
    int* m_output_idx_device;
    float* m_output_conf_device;
    int m_output_obj_area;
    int m_output_obj_area_bi;

    float* m_output_src_transpose_device;

    /// <segmentation>
    int m_output_seg_area;
    int m_output_seg_w;
    int m_output_seg_h;
    float* p_seg;
    cv::Mat proto_buffer;
    cv::Mat mask;
    cv::Mat proto_mask;
};
