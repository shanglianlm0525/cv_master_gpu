#include "include/cv_gpu_alg_cls.h"
#include "include/decode_yolo.h"


/*
CvGpuAlgCls::CvGpuAlgCls(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data)
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
    m_param.num_class = m_output_dims.d[1];

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
    m_output_objects_host = nullptr;
    checkRuntime(cudaMalloc(&m_output_src_device, m_param.batch_size * m_param.num_class * sizeof(float)));
    m_output_objects_host = new float[m_param.batch_size * m_param.num_class];
}
*/

void CvGpuAlgCls::init(InPara& iparam, std::string model_path, std::vector<unsigned char>& model_data)
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
    m_param.num_class = m_output_dims.d[1];

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
    m_output_objects_host = nullptr;
    checkRuntime(cudaMalloc(&m_output_src_device, m_param.batch_size * m_param.num_class * sizeof(float)));
    m_output_objects_host = new float[m_param.batch_size * m_param.num_class];
}


CvGpuAlgCls::~CvGpuAlgCls()
{
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
    delete[] m_output_objects_host;
}


void CvGpuAlgCls::preprocess(std::vector<cv::Mat> imgs)
{
    assert(imgsBatch.size() <= m_param.batch_size);

    if (m_param.fixed_shape)
    {
        // update 20230302, faster. 
        // 1. move uint8_to_float in cuda kernel function. For 8*3*1920*1080, cost time 15ms -> 3.9ms
        // 2. Todo
        unsigned char* pi = m_input_src_device;
        for (size_t i = 0; i < imgs.size(); i++)
        {
            assert(m_param.nOrgHeight == imgs[i].rows and m_param.nOrgWidth = imgs[i].cols);
            checkRuntime(cudaMemcpy(pi, imgs[i].data, sizeof(unsigned char) * m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth, cudaMemcpyHostToDevice));
            pi += m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth;
        }
        resizeDevice(m_param.batch_size, m_input_src_device, m_param.nOrgWidth, m_param.nOrgHeight, m_param.nChannel,
            m_input_resize_device, m_param.nWidth, m_param.nHeight, 114, m_dst2src);
    }
    else {
        float* ri = m_input_resize_device;

        for (auto imgb : imgs) {
            m_param.nOrgHeight = imgb.rows;
            m_param.nOrgWidth = imgb.cols;
            computeAffineMat(m_param, m_dst2src);
            checkRuntime(cudaMalloc(&m_input_src_device, m_param.nChannel * m_param.nOrgHeight * m_param.nOrgWidth * sizeof(unsigned char)));
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

void CvGpuAlgCls::postprocess(std::vector<cv::Mat> imgsBatch)
{
    softmaxDevice(m_output_src_device, m_param.batch_size, m_param.num_class);
    checkRuntime(cudaMemcpy(m_output_objects_host, m_output_src_device, m_param.batch_size * m_param.num_class * sizeof(float), cudaMemcpyDeviceToHost));
}


int CvGpuAlgCls::infer(std::vector<cv::Mat> imgs, DetectInfo* imgsInfoList, ModelPara mPara, DetectInfo* dInfoList, int* dNum)
{
    // assert(imgsBatch.size() <= m_param.batch_size);

    int img_num = imgs.size();

    int obj_num = 0; 
    std::vector<cv::Mat> imgs_batch;
    for (j = 0; j < img_num; j = j + m_param.batch_size)
    {
        imgs_batch.assign(imgs.begin()+ j * m_param.batch_size, imgs.begin() + std::min(img_num, (j + 1) * m_param.batch_size));

        preprocess(imgs_batch);
        float* bindings[] = { m_input_hwc_device, m_output_src_device };
        bool context = m_context->executeV2((void**)bindings);
        if (context == false) {
            return -1;
        }
        postprocess(imgs_batch);

        for (i = j; i < std::min(img_num, j + m_param.batch_size); i++) {

            const float* buffer = m_output_objects_host + m_param.num_class * i;
            auto predict = std::max_element(buffer, buffer + m_param.num_class) - buffer;

            if (predict) {
                // std::cout << "The prediction digit is:" << predict << " " << buffer[1 - predict] << " " << buffer[predict] << std::endl;

                if (buffer[predict] >= mPara.score_threshold and obj_num < mPara.nMaxObjNum - 1)
                {
                    imgsInfoList[i].cls_score = buffer[predict];
                    imgsInfoList[i].cls_id = predict;

                    *(dInfoList + obj_num) = imgsInfoList[i];
                    // (*(dInfoList + obj_num)).conf = prob[predict];
                    obj_num = obj_num + 1;
                }
                else {
                    imgsInfoList[i].cls_score = 0;
                    imgsInfoList[i].cls_id = 0;
                }
            }
            else {
                imgsInfoList[i].cls_score = 0;
                imgsInfoList[i].cls_id = 0;
            }

        }
    }

    *dNum = obj_num;

    return obj_num;
}


