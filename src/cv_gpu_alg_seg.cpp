#include "include/cv_gpu_alg_seg.h"
#include "include/decode_yolo.h"

/*
CvGpuAlgSeg::CvGpuAlgSeg(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data)
{
    m_param = iparam;

    // load data
    model_data = loadModel(model_path);
    this->m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (this->m_runtime == nullptr)
    {
        return;
    }

    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(this->m_runtime->deserializeCudaEngine(model_data.data(), model_data.size()));
    if (this->m_engine == nullptr)
    {
        return;
    }

    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr)
    {
        return;
    }

    // obtain the number of input and output layers
    assert(this->m_engine->getNbBindings() / 2 == 1);
    assert(this->m_engine->getNbBindings() / 2 == 1);

    // input dims
    m_input_dims = this->m_engine->getBindingDimensions(0);
    m_param.batch_size = m_input_dims.d[0];
    m_param.nChannel = m_input_dims.d[1];
    m_param.nHeight = m_input_dims.d[2];
    m_param.nWidth = m_input_dims.d[3];

    // output dims
    m_output_dims = this->m_context->getBindingDimensions(1);
    m_param.num_class = m_output_dims.d[1] - 4;
    m_total_objects = m_output_dims.d[2];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1;
    for (i = 1; i < m_output_dims.nbDims; ++i)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }

    // input cuda malloc
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_rgb_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    if (m_param.fixed_shape)
    {
        assert(m_param.nOrgHeight > 0 and m_param.nOrgWidth > 0);
        checkRuntime(cudaMalloc(&m_input_src_device, m_param.batch_size * m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth * sizeof(unsigned char)));
        computeAffineMat(m_param, m_dst2src);
    }
    checkRuntime(cudaMalloc(&m_input_resize_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));
    if (m_param.nChannel == 3)
    {
        checkRuntime(cudaMalloc(&m_input_rgb_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));
    }
    checkRuntime(cudaMalloc(&m_input_norm_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_hwc_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));


    // output cuda malloc
    m_output_src_device = nullptr;
    m_output_objects_device = nullptr;
    m_output_objects_host = nullptr;
    m_output_objects_width = 7;
    m_output_idx_device = nullptr;
    m_output_conf_device = nullptr;
    m_output_src_transpose_device = nullptr;
    int output_objects_size = m_param.batch_size * (1 + m_param.topK * m_output_objects_width); // 1: count
    checkRuntime(cudaMalloc(&m_output_objects_device, output_objects_size * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_idx_device, m_param.batch_size * m_param.topK * sizeof(int)));
    checkRuntime(cudaMalloc(&m_output_conf_device, m_param.batch_size * m_param.topK * sizeof(float)));
    m_output_objects_host = new float[output_objects_size];

    checkRuntime(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_src_transpose_device, m_param.batch_size * m_output_area * sizeof(float)));
}
*/

void CvGpuAlgSeg::init(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data)
{
    m_param = iparam;

    // load data
    model_data = loadModel(model_path);
    this->m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (this->m_runtime == nullptr)
    {
        return;
    }

    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(this->m_runtime->deserializeCudaEngine(model_data.data(), model_data.size()));
    if (this->m_engine == nullptr)
    {
        return;
    }

    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr)
    {
        return;
    }

    // obtain the number of input and output layers
    assert(this->m_engine->getNbBindings() / 2 == 1);
    assert(this->m_engine->getNbBindings() / 2 == 1);


    // input dims
    m_input_dims = this->m_engine->getBindingDimensions(0);
    m_param.batch_size = m_input_dims.d[0];
    m_param.nChannel = m_input_dims.d[1];
    m_param.nHeight = m_input_dims.d[2];
    m_param.nWidth = m_input_dims.d[3];

    // output dims
    m_output_dims = this->m_context->getBindingDimensions(2);
    m_total_objects = m_output_dims.d[2];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1; // 116 * 8400
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }

    m_output_seg_dims = this->m_context->getBindingDimensions(1);
    assert(m_param.batch_size <= m_output_seg_dims.d[0]);
    m_output_seg_area = 1; // 32 * 160 * 160
    for (int i = 1; i < m_output_seg_dims.nbDims; i++)
    {
        if (m_output_seg_dims.d[i] != 0)
        {
            m_output_seg_area *= m_output_seg_dims.d[i];
        }
    }

    m_output_objects_width = 39;
    m_output_src_width = m_output_dims.d[1]; // 116
    m_output_seg_w = m_output_seg_dims.d[2] * m_output_seg_dims.d[3]; // 160 * 160
    m_output_seg_h = m_output_seg_dims.d[1]; // 32
    m_output_obj_area = 1 + m_param.topK * m_output_objects_width;

      // input cuda malloc
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_rgb_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    if (m_param.fixed_shape)
    {
        assert(m_param.nOrgHeight > 0 and m_param.nOrgWidth > 0);
        checkRuntime(cudaMalloc(&m_input_src_device, m_param.batch_size * m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth * sizeof(unsigned char)));
        computeAffineMat(m_param, m_dst2src);
    }
    checkRuntime(cudaMalloc(&m_input_resize_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));
    if (m_param.nChannel == 3)
    {
        checkRuntime(cudaMalloc(&m_input_rgb_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));
    }
    checkRuntime(cudaMalloc(&m_input_norm_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_hwc_device, m_param.batch_size * m_param.nChannel * m_param.nHeight * m_param.nWidth * sizeof(float)));


    // output cuda malloc
    m_output_objects_device = nullptr;
    m_output_objects_host = nullptr;
    m_output_idx_device = nullptr;
    m_output_conf_device = nullptr;
    m_output_seg_device = nullptr;
    m_output_src_device = nullptr;
    m_output_src_transpose_device = nullptr;
    int output_objects_size = m_param.batch_size * (1 + m_param.topK * m_output_objects_width); // 1: count
    checkRuntime(cudaMalloc(&m_output_objects_device, output_objects_size * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_idx_device, m_param.batch_size * m_param.topK * sizeof(int)));
    checkRuntime(cudaMalloc(&m_output_conf_device, m_param.batch_size * m_param.topK * sizeof(float)));
    m_output_objects_host = new float[output_objects_size];
    checkRuntime(cudaMalloc(&m_output_seg_device, m_param.batch_size * m_output_seg_area * sizeof(float)));
    m_output_seg_host = new float[m_param.batch_size * m_output_seg_area];

    checkRuntime(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_src_transpose_device, m_param.batch_size * m_output_area * sizeof(float)));

}


CvGpuAlgSeg::~CvGpuAlgSeg()
{
    checkRuntime(cudaFree(m_output_src_transpose_device));

    // input
    if (m_param.fixed_shape)
    {
        checkRuntime(cudaFree(m_input_src_device));
    }
    checkRuntime(cudaFree(m_input_resize_device));
    if (m_param.nChannel == 3)
    {
        checkRuntime(cudaFree(m_input_rgb_device));
    }
    checkRuntime(cudaFree(m_input_norm_device));
    checkRuntime(cudaFree(m_input_hwc_device));
    // output
    checkRuntime(cudaFree(m_output_src_device));
    checkRuntime(cudaFree(m_output_objects_device));
    checkRuntime(cudaFree(m_output_seg_device));
    checkRuntime(cudaFree(m_output_idx_device));
    checkRuntime(cudaFree(m_output_conf_device));
    delete[] m_output_objects_host;
    delete[] m_output_seg_host;
}


void CvGpuAlgSeg::preprocess(std::vector<cv::Mat> imgsBatch)
{
    assert(imgsBatch.size() <= m_param.batch_size);

    if (m_param.fixed_shape)
    {
        // update 20230302, faster. 
        // 1. move uint8_to_float in cuda kernel function. For 8*3*1920*1080, cost time 15ms -> 3.9ms
        // 2. Todo
        unsigned char* pi = m_input_src_device;
        for (size_t i = 0; i < imgsBatch.size(); i++)
        {
            assert(m_param.nOrgHeight = imgsBatch[i].rows and m_param.nOrgWidth = imgsBatch[i].cols);
            checkRuntime(cudaMemcpy(pi, imgsBatch[i].data, sizeof(unsigned char) * m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth, cudaMemcpyHostToDevice));
            pi += m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth;
        }
        resizeDevice(m_param.batch_size, m_input_src_device, m_param.nOrgWidth, m_param.nOrgHeight, m_param.nChannel,
            m_input_resize_device, m_param.nWidth, m_param.nHeight, 114, m_dst2src);
    }
    else {
        float* ri = m_input_resize_device;

        for (auto imgb : imgsBatch) {
            m_param.nOrgHeight = imgb.rows;
            m_param.nOrgWidth = imgb.cols;
            computeAffineMat(m_param, m_dst2src);
            checkRuntime(cudaMalloc(&m_input_src_device, 1 * m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth * sizeof(unsigned char)));
            checkRuntime(cudaMemcpy(m_input_src_device, imgb.data, sizeof(unsigned char) * 1 * m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth, cudaMemcpyHostToDevice));

            // use gpu do resize operate
            resizeDevice(1, m_input_src_device, m_param.nOrgWidth, m_param.nOrgHeight, m_param.nChannel,
                ri, m_param.nWidth, m_param.nHeight, 114, m_dst2src);

            ri += m_param.nChannel * m_param.nHeight * m_param.nWidth;

            checkRuntime(cudaFree(m_input_src_device));
        }
    }

    if (m_param.nChannel == 3)
    {
        bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.nWidth, m_param.nHeight, m_param.nChannel,
            m_input_rgb_device, m_param.nWidth, m_param.nHeight);
        normDevice(m_param.batch_size, m_input_rgb_device, m_param.nWidth, m_param.nHeight, m_param.nChannel,
            m_input_norm_device, m_param.nWidth, m_param.nHeight, m_param);
    }
    else {
        normDevice(m_param.batch_size, m_input_resize_device, m_param.nWidth, m_param.nHeight, m_param.nChannel,
            m_input_norm_device, m_param.nWidth, m_param.nHeight, m_param);
    }
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.nWidth, m_param.nHeight, m_param.nChannel,
        m_input_hwc_device, m_param.nWidth, m_param.nHeight);
}

void CvGpuAlgSeg::postprocess(std::vector<cv::Mat> imgsBatch)
{
    transposeDevice(m_param, m_output_src_device, m_total_objects, m_output_src_width, m_total_objects * m_output_src_width,
        m_output_src_transpose_device, m_output_src_width, m_total_objects);
    decodeDevice(m_param, m_output_src_transpose_device, m_output_src_width, m_total_objects, m_output_area,
        m_output_objects_device, m_output_objects_width, m_param.topK, 1);
    nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.nChannel, m_output_obj_area);
    // nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width, m_param.topK,  m_param.nChannel, m_output_obj_area, m_output_idx_device, m_output_conf_device);
    checkRuntime(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * m_output_obj_area, cudaMemcpyDeviceToHost));
    checkRuntime(cudaMemcpy(m_output_seg_host, m_output_seg_device, m_param.batch_size * sizeof(float) * m_output_seg_area, cudaMemcpyDeviceToHost));
}


float sigmoid_function(float a) {
    return 1. / (1. + exp(-a));
}

int CvGpuAlgSeg::infer(std::vector<cv::Mat> imgsBatch, ModelPara mPara, DetectInfo* dInfoList, int* dNum)
{
    assert(imgsBatch.size() <= m_param.batch_size);

    // reset
    checkRuntime(cudaMemset(m_output_objects_device, 0, sizeof(float) * m_param.batch_size * (1 + m_output_objects_width * m_param.topK)));

    float* bindings[] = { m_input_hwc_device, m_output_seg_device, m_output_src_device };
    bool context = m_context->executeV2((void**)bindings);
    if (context == false) {
        return -1;
    }

    postprocess(imgsBatch);


    *dNum = 0;
    int num_boxes = 0;
    float* ptr;
    float x_lt, y_lt, x_rb, y_rb;
    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        num_boxes = std::min((int)(m_output_objects_host + bi * m_output_obj_area)[0], m_param.topK);
        p_seg = m_output_seg_host + bi * m_output_seg_area;
        proto_buffer = cv::Mat(m_output_seg_h, m_output_seg_w, CV_32F, p_seg);
        m_output_obj_area_bi = bi * m_output_obj_area;
        for (size_t i = 0; i < num_boxes; i++)
        {
            ptr = m_output_objects_host + m_output_obj_area_bi + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag)
            {
                if (*dNum < mPara.nMaxObjNum - 1) {
                    x_lt = m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2;
                    y_lt = m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
                    x_rb = m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2;
                    y_rb = m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;

                    // seg 
                    mask = cv::Mat(1, m_output_seg_h, CV_32F, p_seg + 7);
                    proto_mask = mask * proto_buffer;
                    for (int col = 0; col < proto_mask.cols; col++) {
                        proto_mask.at<float>(0, col) = sigmoid_function(proto_mask.at<float>(0, col));
                    }
                    proto_mask = proto_mask.reshape(1, m_output_seg_dims.d[2]); // 1x25600 -> 160x160
                    cv::resize(proto_mask, proto_mask, cv::Size(m_param.nOrgWidth, m_param.nOrgHeight));

                    if (dInfo.seg_num > dInfo.height * dInfo.width) {
                        dInfo.seg_num = 0;
                        for (int r = dInfo.y; r < dInfo.y + dInfo.height; r++) {
                            for (int c = dInfo.x; c < dInfo.x + dInfo.width; c++) {
                                if (proto_mask.at<float>(r, c) > 0.5) {
                                    proto_mask.at<float>(r, c) = proto_mask.at<float>(r, c) * 255.0;
                                    dInfo.seg_data[dInfo.seg_num] = 1;
                                }
                                else {
                                    proto_mask.at<float>(r, c) = 0.0;
                                    dInfo.seg_data[dInfo.seg_num] = 0;
                                }
                                dInfo.seg_num = dInfo.seg_num + 1;
                            }
                        }
                    }
                    cv::imwrite("proto_mask_"+std::to_string(i) + ".png", proto_mask);

                    dInfo.img_idx = bi;
                    dInfo.det_id = (int)ptr[5];
                    dInfo.det_score = ptr[4];

                    dInfo.x = std::max(floor(x_lt), 0.f);
                    dInfo.y = std::max(floor(y_lt), 0.f);
                    dInfo.width = floor(x_rb - x_lt);
                    dInfo.height = floor(y_rb - y_lt);
                    if (dInfo.width < 1 or dInfo.height < 1) {
                        dInfo.width = std::max(dInfo.width, 1);
                        dInfo.height = std::max(dInfo.height, 1);
                    }
                    *(dInfoList + *dNum) = dInfo;
                    *dNum += 1;
                }

            }
        }
    }

    return *dNum;
}
