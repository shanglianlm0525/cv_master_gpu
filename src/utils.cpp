#include "include/utils.h"


void saveBinaryFile(float* vec, size_t len, const std::string& file)
{
	std::ofstream  out(file, std::ios::out | std::ios::binary);
	if (!out.is_open())
		return;
	out.write((const char*)vec, sizeof(float) * len);
	out.close();
}

std::vector<uint8_t> readBinaryFile(const std::string& file)
{

	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
		return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}


std::vector<unsigned char> loadModel(const std::string& file)
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		return {};
	}
	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}




void computeAffineMat(InPara m_param, AffineMat& m_dst2src) {

	float a = float(m_param.nHeight) / m_param.nOrgHeight;
	float b = float(m_param.nWidth) / m_param.nOrgWidth;
	float scale = a < b ? a : b;
	cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.nOrgWidth + m_param.nWidth + scale - 1) * 0.5,
		0.f, scale, (-scale * m_param.nOrgHeight + m_param.nHeight + scale - 1) * 0.5);
	cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
	cv::invertAffineTransform(src2dst, dst2src);

	m_dst2src.v0 = dst2src.ptr<float>(0)[0];
	m_dst2src.v1 = dst2src.ptr<float>(0)[1];
	m_dst2src.v2 = dst2src.ptr<float>(0)[2];
	m_dst2src.v3 = dst2src.ptr<float>(1)[0];
	m_dst2src.v4 = dst2src.ptr<float>(1)[1];
	m_dst2src.v5 = dst2src.ptr<float>(1)[2];
}


DeviceTimer::DeviceTimer()
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
}

float DeviceTimer::getUsedTime()
{
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

DeviceTimer::DeviceTimer(cudaStream_t stream)
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, stream);
}

float DeviceTimer::getUsedTime(cudaStream_t stream)
{
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

DeviceTimer::~DeviceTimer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}