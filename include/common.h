#pragma once
#pragma warning(disable:4996)

// #define USING_DECRYPT_MODEL

#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


const uint8_t nKey = 1295;

// const char g_key[17] = "asdfwetyhjuytrfd"; // StringingVIDLL
// const char g_key[17] = "mnlkjpoiuyhbvgtf"; // delivery_ocr
// const char g_key[17] = "hsgidjwgsdgsdfgs"; // StringingADDLL
const char g_key[17] = "www.bot8rong.com";
const char g_iv[17] = "gfdertfghjkuyrtg";//ECB MODE不需要关心chain，可以填空


void read_file(std::string sfile, std::vector<uint8_t>& sdata);


void decrypt_file(std::vector<uint8_t>& model_data);


void softmax(const float* input, float* output, int n);


std::string get_unique_id();

int get_dongle_id();

int decryptString(std::string str);


// AES加密
std::string EncryptionAES(std::string strSrc);

// AES解密
std::string DecryptionAES(std::string strSrc);


std::vector<std::string> split_str(const std::string& str, char delimiter);


int verification_localTime(std::string setTime);