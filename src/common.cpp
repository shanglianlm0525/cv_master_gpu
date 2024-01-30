
#include "include/common.h"
#include "include/AES.h"
#include "include/Base64.h"
#include "include/IpeLicInfo.h"

std::vector<uint8_t> read_file(const std::string& file_name)
{
	int begin, end;
	std::ifstream file(file_name, std::ios::in | std::ios::binary);
	if (!file)
	{
		exit(-1);
	}

	begin = file.tellg();
	file.seekg(0, std::ios::end);
	end = file.tellg();
	size_t len = end - begin;


	std::vector<uint8_t> img_bytes(len);
	file.seekg(0, std::ios::beg);
	file.read(reinterpret_cast<char*>(img_bytes.data()), len);
	return img_bytes;
}

void read_file(std::string sfile, std::vector<uint8_t>& sdata)
{
	int begin, end;
	std::ifstream file;

	// model file
	file.open(sfile, std::ios::in | std::ios::binary);
	if (!file)
	{
		exit(-1);
	}

	begin = file.tellg();
	file.seekg(0, std::ios::end);
	end = file.tellg();
	size_t len = end - begin;

	sdata.resize(len);
	file.seekg(0, std::ios::beg);
	file.read(reinterpret_cast<char*>(sdata.data()), len);
	file.close();
}


void decrypt_file(std::vector<uint8_t>& model_data)
{
	// decrypt model_data
	for (int i = 0; i < model_data.size(); ++i)
	{
		model_data[i] = model_data[i] - nKey;
	}

}


void softmax(std::vector<float>& input) {
	float maxn = 0.0;
	float sum = 0.0;
	maxn = *max_element(input.begin(), input.end());
	std::for_each(input.begin(), input.end(), [maxn, &sum](float& d) {
		d = exp(d - maxn); sum += d; }); 	//cmath c11
	std::for_each(input.begin(), input.end(), [sum](float& d) { d = d / sum; });
	return;
}

void softmax(const float* input, float* output, int n) {
	for (int i = 0; i < n; ++i)
	{
		output[i] = input[i];
	}

	float maxn = 0.0;
	float sum = 0.0;
	maxn = *std::max_element(output, output + n);
	std::for_each(output, output + n, [maxn, &sum](float& d) {d = exp(d - maxn); sum += d; }); 	//cmath c11
	std::for_each(output, output + n, [sum](float& d) { d = d / sum; });
}


std::string get_unique_id() {
	std::string result;

	// 尝试获取UUID  
	std::string uuid_output = "";
	std::string command = "wmic csproduct get UUID";
	FILE* pipe = _popen(command.c_str(), "r");
	if (pipe) {
		char buffer[128];
		while (!feof(pipe)) {
			if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
				uuid_output += buffer;
			}
		}
		_pclose(pipe);
	}

	// 如果UUID无效，尝试获取主板序列号  
	if (uuid_output.empty() || uuid_output.length() < 10) {
		std::string baseboard_output = "";
		command = "wmic baseboard get serialnumber";
		pipe = _popen(command.c_str(), "r");
		if (pipe) {
			char buffer[128];
			while (!feof(pipe)) {
				if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
					baseboard_output += buffer;
				}
			}
			_pclose(pipe);
		}
		if (!baseboard_output.empty() || uuid_output == "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF" || uuid_output == "Default string") {
			result = baseboard_output;
		}
		else {
			// 如果主板序列号也无效，尝试获取硬盘序列号  
			std::string disk_output = "";
			command = "wmic diskdrive get serialnumber";
			pipe = _popen(command.c_str(), "r");
			if (pipe) {
				char buffer[128];
				while (!feof(pipe)) {
					if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
						disk_output += buffer;
					}
				}
				_pclose(pipe);
			}
			if (!disk_output.empty()) {
				result = disk_output;
			}
		}
	}
	else {
		result = uuid_output;
	}

	size_t pos = result.find_first_of('\n');
	if (pos != std::string::npos) {
		result.erase(0, pos + 1);
	}
	pos = result.find_first_of(' ');
	if (pos != std::string::npos) {
		result.erase(pos);
	}

	return result;
}

std::string encryptString(int num) {

	std::vector<std::string> hexDigits;
	while (num > 0) {
		int digit = num % 10;
		std::stringstream ss;
		ss << std::setfill('F') << std::setw(4) << std::hex << (digit);
		hexDigits.push_back(ss.str());
		num /= 10;
	}

	std::stringstream result;
	for (int i = hexDigits.size() - 1; i >= 0; i--) {
		if (i == 0) {
			result << hexDigits[i];
		}
		else {
			result << hexDigits[i] << "-";
		}
	}

	return result.str();
}

int decryptString(std::string str) {

	std::string deStr = str;

	deStr.erase(std::remove(deStr.begin(), deStr.end(), '-'), deStr.end());
	deStr.erase(std::remove(deStr.begin(), deStr.end(), 'F'), deStr.end());

	return std::stoi(deStr);
}

int get_dongle_id() {
	CIpeLicDevInfo info;
	info.ScanHardware();
	int devCount = info.getDeviceCount();
	IpeLicDevInfo* infoBuff = info.getDeviceInfoBuff();

	for (int n(0); n < devCount; n++)
	{
		infoBuff[n].type;  // device type: 0 = Unknown; 1=Dongle;2=Board;3=System
		infoBuff[n].number;// device number

		// std::cout << infoBuff[n].type << " " << infoBuff[n].number << std::endl;
		if (infoBuff[n].type == 1) {
			// std::string encryptedString = encryptString(std::to_string(infoBuff[n].number));
			return infoBuff[n].number;
		}
	}

	return -9999;
}

std::string EncryptionAES(std::string strSrc) //AES加密
{
	size_t length = strSrc.length();
	int block_num = length / BLOCK_SIZE + 1;
	//明文
	char* szDataIn = new char[block_num * BLOCK_SIZE + 1];
	memset(szDataIn, 0x00, block_num * BLOCK_SIZE + 1);
	strcpy(szDataIn, strSrc.c_str());

	//进行PKCS7Padding填充。
	int k = length % BLOCK_SIZE;
	int j = length / BLOCK_SIZE;
	int padding = BLOCK_SIZE - k;
	for (int i = 0; i < padding; i++)
	{
		szDataIn[j * BLOCK_SIZE + k + i] = padding;
	}
	szDataIn[block_num * BLOCK_SIZE] = '\0';

	//加密后的密文
	char* szDataOut = new char[block_num * BLOCK_SIZE + 1];
	memset(szDataOut, 0, block_num * BLOCK_SIZE + 1);

	//进行进行AES的CBC模式加密
	AES aes;
	aes.MakeKey(g_key, g_iv, 16, 16);
	aes.Encrypt(szDataIn, szDataOut, block_num * BLOCK_SIZE, AES::CBC);
	std::string str = base64_encode((unsigned char*)szDataOut,
		block_num * BLOCK_SIZE);
	delete[] szDataIn;
	delete[] szDataOut;

	return str;
}

std::string DecryptionAES(std::string strSrc) //AES解密
{
	std::string strData = base64_decode(strSrc);
	size_t length = strData.length();
	//密文
	char* szDataIn = new char[length + 1];
	memcpy(szDataIn, strData.c_str(), length + 1);
	//明文
	char* szDataOut = new char[length + 1];
	memcpy(szDataOut, strData.c_str(), length + 1);

	//进行AES的CBC模式解密
	AES aes;
	aes.MakeKey(g_key, g_iv, 16, 16);
	aes.Decrypt(szDataIn, szDataOut, length, AES::CBC);

	//去PKCS7Padding填充
	if (0x00 < szDataOut[length - 1] <= 0x16)
	{
		int tmp = szDataOut[length - 1];
		for (int i = length - 1; i >= length - tmp; i--)
		{
			if (szDataOut[i] != tmp)
			{
				memset(szDataOut, 0, length);
				break;
			}
			else
				szDataOut[i] = 0;
		}
	}
	std::string strDest(szDataOut);
	delete[] szDataIn;
	delete[] szDataOut;
	return strDest;
}


std::vector<std::string> split_str(const std::string& str, char delimiter) {
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(str);

	while (std::getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}

	return tokens;
}


int verification_localTime(std::string setTime) {

	// 获取当前时间
	std::time_t currentTime = std::time(nullptr);
	std::tm* localTime = std::localtime(&currentTime);

	// 比较时间差距
	std::tm setTimeStruct = {};
	std::istringstream ss(setTime);
	ss >> std::get_time(&setTimeStruct, "%Y-%m-%d");
	std::time_t setTimeT = std::mktime(&setTimeStruct);

	double diffSeconds = std::difftime(setTimeT, currentTime);

	return diffSeconds < 0.0;
}