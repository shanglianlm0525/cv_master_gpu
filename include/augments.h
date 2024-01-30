#pragma once

struct ModelConfig {
	const char* model_path;   //  model_path
	const char* weight_path;  //  weight_path
	int max_batch_size = 1;   // batch_size
	int async_num = 0;
};

struct ModelPara {
	float score_threshold = 0.2;     //  score threshold
	float nms_threshold = 0.4;       // threshold for NMS

	// min
	float min_score_threshold = 0;
	float min_nms_threshold = 0;

	// max 
	float max_score_threshold = 1;
	float max_nms_threshold = 1;

	// flag
	int flag = 1; // 0: not use  1: use
	int nMaxObjNum;
};


struct DetectInfo {
	int img_idx;
	int x; //!< x coordinate of the top-left corner
	int y; //!< y coordinate of the top-left corner
	int width; //!< width of the rectangle
	int height; //!< height of the rectangle;
	// det
	float det_score;
	int det_id;
	// cls
	float cls_score;
	int cls_id;
	// seg
	char* seg_data;
	int seg_num;
};