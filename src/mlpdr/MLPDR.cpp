/*
 * LPRDetector.cpp
 *
 *  Created on: Jun 25, 2019
 *      Author: lqian
 */

#include "mlpdr/MLPDR.h"
#include "mlpdr/label.hpp"

#include <Interpreter.hpp>
#include <MNNDefine.h>
#include <Tensor.hpp>
#include <ImageProcess.hpp>

namespace mlpdr {

using namespace std;
using namespace MNN;
using namespace MNN::CV;
using namespace cv;

std::shared_ptr<Interpreter> LPRNet = NULL;
Session * ocr_session = nullptr;
Tensor * ocr_input = nullptr;
Tensor * ocr_output = nullptr;

std::shared_ptr<MNN::Interpreter> LPCNet_ = NULL;
Session * lpc_session = nullptr;
Tensor * lpc_input = nullptr;
Tensor * lpc_output = nullptr;


std::shared_ptr<MNN::Interpreter> PNet_ = NULL;
std::shared_ptr<MNN::Interpreter> RNet_ = NULL;
std::shared_ptr<MNN::Interpreter> ONet_ = NULL;


MNN::Session * sess_p = NULL;
MNN::Session * sess_r = NULL;
MNN::Session * sess_o = NULL;

MNN::Tensor * p_input = nullptr;
MNN::Tensor * p_out_pro = nullptr;
MNN::Tensor * p_out_reg = nullptr;

MNN::Tensor * r_input = nullptr;
MNN::Tensor * r_out_pro = nullptr;
MNN::Tensor * r_out_reg = nullptr;

MNN::Tensor * o_input = nullptr;
MNN::Tensor * o_out_pro = nullptr;
MNN::Tensor * o_out_reg = nullptr;
MNN::Tensor * o_out_lank = nullptr;

std::shared_ptr<ImageProcess> pretreat_data;
std::shared_ptr<ImageProcess> lpr_pretreat_data, lpc_pretreat_data;

std::vector<PlateInfo> candidate_boxes_;
std::vector<PlateInfo> total_boxes_;

static float threhold_p = 0.7f;
static float threhold_r = 0.8f;
static float threhold_o = 0.8f;
static float iou_threhold = 0.7f;
static float factor = 0.709f;
//static int min_face = 48;

//pnet config
static const float pnet_stride = 2;
static const float pnet_cell_size_height = 12;
static const float pnet_cell_size_width = 30;
static const int pnet_max_detect_num = 5000;
//mean & std
static const float mean_val = 127.5f;
static const float std_val = 0.0078125f;

static bool CompareBBox(const PlateInfo & a, const PlateInfo & b) {
	return a.bbox.score > b.bbox.score;
}

static float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_,
		float ymin_, float xmax_, float ymax_, bool is_iom) {
	float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
	float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
	if (iw <= 0 || ih <= 0)
		return 0;
	float s = iw * ih;
	if (is_iom) {
		float ov = s / std::min((xmax - xmin + 1) * (ymax - ymin + 1), 	(xmax_ - xmin_ + 1) * (ymax_ - ymin_ + 1));
		return ov;
	} else {
		float ov = s / ((xmax - xmin + 1) * (ymax - ymin + 1) + (xmax_ - xmin_ + 1) * (ymax_ - ymin_ + 1) - s);
		return ov;
	}
}

static std::vector<PlateInfo> NMS(std::vector<PlateInfo>& bboxes, float thresh,
		char methodType) {
	std::vector<PlateInfo> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}
		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		FaceBox select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin
				+ 1) * (select_bbox.ymax - select_bbox.ymin + 1));
		float x1 = static_cast<float>(select_bbox.xmin);
		float y1 = static_cast<float>(select_bbox.ymin);
		float x2 = static_cast<float>(select_bbox.xmax);
		float y2 = static_cast<float>(select_bbox.ymax);

		select_idx++;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			FaceBox & bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x  + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y  + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1)
					* (bbox_i.ymax - bbox_i.ymin + 1));
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (static_cast<float>(area_intersect)
						/ (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (static_cast<float>(area_intersect) / std::min(area1, area2)
				> thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}
static void BBoxRegression(vector<PlateInfo>& bboxes) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float *bbox_reg = bboxes[i].bbox_reg;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		bbox.xmin += bbox_reg[0] * w;
		bbox.ymin += bbox_reg[1] * h;
		bbox.xmax += bbox_reg[2] * w;
		bbox.ymax += bbox_reg[3] * h;

		//		bbox.xmax += bbox_reg[2] * w;
		//		bbox.ymax += bbox_reg[3] * h;
	}
}
static void BBoxPad(vector<PlateInfo>& bboxes, int width, int height) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		bbox.xmin = round(std::max(bbox.xmin, 1.0f));
		bbox.ymin = round(std::max(bbox.ymin, 1.0f));
		bbox.xmax = round(std::min(bbox.xmax, width - 1.f));
		bbox.ymax = round(std::min(bbox.ymax, height - 1.f));
	}
}
static void BBoxPadSquare(vector<PlateInfo>& bboxes, int width, int height) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		float side = h > w ? h : w;
		float side_w = h>w ? h:w;
		float side_h = 12 / 30.0 * side_w;  // 24,60, 48,120

		bbox.xmin = round(std::max(bbox.xmin + (w - side_w) * 0.5f, 0.f));
		bbox.ymin = round(std::max(bbox.ymin + (h - side_h) * 0.5f, 0.f));
		bbox.xmax = round(std::min(bbox.xmin + side_w - 1,   width - 1.f));
		bbox.ymax = round(std::min(bbox.ymin + side_h  - 1,  height - 1.f));
	}
}
static void GenerateBBox(float * prob1_confidence, float *reg_box,
		int feature_map_w_, int feature_map_h_, float scale, float thresh) {
	candidate_boxes_.clear();
	int spatical_size = feature_map_w_ * feature_map_h_;
	for (int h=0; h<feature_map_h_; h++) {
		for (int w=0; w < feature_map_w_; w++, prob1_confidence++) {
			if (* prob1_confidence > thresh) {
				PlateInfo faceInfo;
				FaceBox & bbox = faceInfo.bbox;
				bbox.score = *prob1_confidence;
				bbox.xmin = w * pnet_stride * 2 / scale;
				bbox.ymin = h * pnet_stride / scale;
				bbox.xmax = ( w * pnet_stride + pnet_cell_size_width + 1 - 1.f) / scale;
				bbox.ymax = ( h * pnet_stride + pnet_cell_size_height + 1 - 1.f) / scale;

				// NxCxHxW format
				int index = h * feature_map_w_ + w;
				faceInfo.bbox_reg[0] = reg_box[index];
				faceInfo.bbox_reg[1] = reg_box[index + spatical_size];
				faceInfo.bbox_reg[2] = reg_box[index + spatical_size * 2];
				faceInfo.bbox_reg[3] = reg_box[index + spatical_size * 3];
				candidate_boxes_.push_back(faceInfo);
			}
		}
	}

	//	float v_scale = 1 / scale;
	//	for (int i = 0; i < spatical_size; ++i) {
	//		int stride = i << 2;
	//		if (confidence_data[stride + 1] >= thresh) {
	//			int y = i / feature_map_w_;
	//			int x = i - feature_map_w_ * y;
	//			FaceInfo faceInfo;
	//			FaceBox &faceBox = faceInfo.bbox;
	//
	//			faceBox.xmin = (float) (x * pnet_stride) * v_scale;
	//			faceBox.ymin = (float) (y * pnet_stride) * v_scale;
	//			faceBox.xmax = (float) (x * pnet_stride + pnet_cell_size_width - 1.f)  * v_scale;
	//			faceBox.ymax = (float) (y * pnet_stride + pnet_cell_size_height - 1.f) * v_scale;
	//
	//			faceInfo.bbox_reg[0] = reg_box[stride];
	//			faceInfo.bbox_reg[1] = reg_box[stride + 1];
	//			faceInfo.bbox_reg[2] = reg_box[stride + 2];
	//			faceInfo.bbox_reg[3] = reg_box[stride + 3];
	//
	//			faceBox.score = confidence_data[stride];
	//			candidate_boxes_.push_back(faceInfo);
	//		}
	//	}
}

MLPDR::MLPDR(const string& proto_model_dir,
		float threhold_p_, float threhold_r_, float threhold_o_ , float factor_) {
	threhold_p = threhold_p_;
	threhold_r = threhold_r_;
	threhold_o = threhold_o_;
	factor = factor_;
	//	threads_num = 2;
	PNet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "/det1.mnn").c_str()));
	RNet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "/det2.mnn").c_str()));
	ONet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "/det3.mnn").c_str()));

	MNN::ScheduleConfig config;
	config.type = MNN_FORWARD_CPU;
	config.numThread = 4; // 1 faster

	// we dont use backend config for x86 testing
	//	    BackendConfig backendConfig;
	//	    backendConfig.precision = BackendConfig::Precision_Low;
	//	    backendConfig.power = BackendConfig::Power_High;
	//	    config.backendConfig = &backendConfig;

	sess_p = PNet_->createSession(config);
	sess_r = RNet_->createSession(config);
	sess_o = ONet_->createSession(config);

	p_input = PNet_->getSessionInput(sess_p, NULL);
	p_out_pro = PNet_->getSessionOutput(sess_p, "prob1");
	p_out_reg = PNet_->getSessionOutput(sess_p, "conv4-2");

	r_input = RNet_->getSessionInput(sess_r, NULL);
	r_out_pro = RNet_->getSessionOutput(sess_r, "prob1");
	r_out_reg = RNet_->getSessionOutput(sess_r, "conv5-2");

//	auto outputs = ONet_->getSessionOutputAll(sess_o);
//	for (auto output: outputs) {
//		printf("ONet output name: %s \n", output.first.c_str());
//				// ,output.second->shape()[1],
//				// output.second->shape()[2],
//				// output.second->shape()[3]
////		);
//	}

	o_input = ONet_->getSessionInput(sess_o, NULL);
	o_out_pro = ONet_->getSessionOutput(sess_o, "prob1");
	o_out_reg = ONet_->getSessionOutput(sess_o, "conv6-2");
	o_out_lank = ONet_->getSessionOutput(sess_o, "conv6-3");



	LPRNet =  shared_ptr<Interpreter>( Interpreter::createFromFile((proto_model_dir + "/lpr.mnn").c_str()));
	ocr_session = LPRNet->createSession(config);
	ocr_input = LPRNet->getSessionInput(ocr_session, NULL);
	ocr_output = LPRNet->getSessionOutput(ocr_session, NULL);
	LPRNet->resizeTensor(ocr_input, ocr_input->shape());
	LPRNet->resizeSession(ocr_session);

	Matrix lpr_trans;
	lpr_trans.setScale(1.0f, 1.0f);
	ImageProcess::Config lpr_config;
	lpr_config.filterType = NEAREST;
	const float mean_vals[3] = { 116.407, 133.722, 124.187 };
	const float norm_vals[3] = { 1.0f, 1.0f, 1.0f };
	::memcpy(lpr_config.mean, mean_vals, sizeof(mean_vals));
	::memcpy(lpr_config.normal, norm_vals, sizeof(norm_vals));
	lpr_config.sourceFormat = RGBA;
	lpr_config.destFormat = BGR;

	lpr_pretreat_data = std::shared_ptr<ImageProcess>(ImageProcess::create(lpr_config));
	lpr_pretreat_data->setMatrix(lpr_trans);

	LPCNet_ = shared_ptr<Interpreter>( Interpreter::createFromFile((proto_model_dir + "/lpc.mnn").c_str()));
	lpc_session = LPCNet_->createSession(config);
	lpc_input = LPCNet_->getSessionInput(lpc_session, NULL);
	lpc_output = LPCNet_->getSessionOutput(lpc_session, NULL);

	Matrix lpc_trans;
	lpc_trans.setScale(1.0f, 1.0f);
	ImageProcess::Config lpc_config;
	lpc_config.filterType = BICUBIC;
	const float lpc_mean_vals[3] = { 89.9372, 81.1989, 73.6352 };
	//	norm_vals[3] = { 1.0f, 1.0f, 1.0f };
	::memcpy(lpc_config.mean, mean_vals, sizeof(lpc_mean_vals));
	::memcpy(lpc_config.normal, norm_vals, sizeof(norm_vals));
	lpc_config.sourceFormat = RGBA;
	lpc_config.destFormat = BGR;
	lpc_pretreat_data = std::shared_ptr<ImageProcess>(ImageProcess::create(lpc_config));
	lpc_pretreat_data->setMatrix(lpc_trans);

	plateColorDict.push_back("白");
	plateColorDict.push_back("黄");
	plateColorDict.push_back("蓝");
	plateColorDict.push_back("黑");
	plateColorDict.push_back("绿");

	printf("initialized!\n");
}

MLPDR::~MLPDR() {
	PNet_->releaseModel();
	RNet_->releaseModel();
	ONet_->releaseModel();
	LPRNet->releaseModel();
}

uint8_t* get_img(cv::Mat img) {
	uchar * colorData = new uchar[img.total() * 4];
	cv::Mat MatTemp(img.size(), CV_8UC4, colorData);
	cv::cvtColor(img, MatTemp, CV_BGR2RGBA, 4);
	return (uint8_t *) MatTemp.data;
}


void ctc_decode(Tensor * output, vector<int> & codes) {
	Tensor outputHost(output, output->getDimensionType());
	output->copyToHostTensor(&outputHost);
	auto values = outputHost.host<float>();
	int prev_class_idx = -1;
	for (int t=0; t<output->batch(); t++) {
		int max_class_idx = 0;
		float max_prob = *values;
		values++;
		for (int c=1; c < output->height(); c++, values++) {
			if (*values > max_prob) {
				max_prob = *values;
				max_class_idx = c;
			}
		}

		if (max_class_idx !=0 && max_class_idx != prev_class_idx) {
			codes.push_back(max_class_idx);
		}
		prev_class_idx = max_class_idx;
	}
}

std::string decode_plateNo(const vector<int> & codes) {
	string plateNo = "";
	for( auto it=codes.begin(); it != codes.end(); ++it) {
		plateNo += label[*it];
	}
	return plateNo;
}

void MLPDR::recognize_plate_infos(const cv::Mat & img, vector<PlateInfo> & plateInfos) {
	for (auto & faceInfo: plateInfos) {
		vector<int> codes = {};
		cv::Point2f srcPoints[4];
		cv::Point2f dstPoints[4];

		int x0 = 0;		int y0 = 0;
		int x1 = 128;	int y1 = 0;
		int x2 = 128;	int y2 = 32;
		int x3 = 0;		int y3 = 32;
		dstPoints[0] = cv::Point2f(x0, y0);
		dstPoints[1] = cv::Point2f(x1, y1);
		dstPoints[2] = cv::Point2f(x2, y2);
		dstPoints[3] = cv::Point2f(x3, y3);

		for (int i=0; i<4; i++) {
			int x = i*2;
			int y = x + 1;
			srcPoints[i] = cv::Point2f(faceInfo.landmark[x], faceInfo.landmark[y]);
		}

		cv::Mat plate = cv::Mat::zeros(32, 128, img.type());
		cv::Mat warp_mat = cv::getAffineTransform(srcPoints, dstPoints);
		cv::warpAffine(img, plate, warp_mat, plate.size(), cv::INTER_LINEAR);

		uint8_t *pImg = get_img(plate);
		lpr_pretreat_data->convert(pImg, plate.cols, plate.rows,  0, ocr_input);
		LPRNet->runSession(ocr_session);
		ctc_decode(ocr_output, codes);
		faceInfo.plateNo = decode_plateNo(codes);
		delete pImg;

		// predict plate color
		cv::resize(plate, plate, Size(110, 22));
		pImg = get_img(plate);
		lpc_pretreat_data->convert(pImg, plate.cols, plate.rows,  0, lpc_input);
		LPCNet_->runSession(lpc_session);
		Tensor lpc_output_host(lpc_output, lpc_output->getDimensionType());
		lpc_output->copyToHostTensor(&lpc_output_host);
		auto * probs = lpc_output_host.host<float>();
		float max = probs[0];
		int clsId = 0;
		for (int i=1; i<5; i++) {
			if (probs[i] > max) {
				max = probs[i];
				clsId = i;
			}
		}
		faceInfo.plateColor = plateColorDict[clsId];
		delete pImg;
	}
}

void fillInput(const std::vector<int> & dim, const cv::Mat & sample, Tensor*  input) {
	int hs = dim[2];
	int ws = dim[3];
	auto inputHost = std::shared_ptr<MNN::Tensor>(MNN::Tensor::create<float>({1, hs, ws, 3}));
	cv::Mat resized;
	if (sample.cols != ws || sample.rows != hs) {
//		printf("resize to %d x %d \n", hs, ws);
		cv::resize(sample, resized, cv::Size(ws, hs), 0, 0, cv::INTER_NEAREST);
	}
	else {
		resized = sample;
	}

	int index=0;
	for (int h=0; h < hs; h++) {
		for (int w=0; w < ws; w++) {
			cv::Vec3f pixel = resized.at<cv::Vec3f>(h, w);
			for (int c = 0 ; c < 3; c++, index++) {
				inputHost->host<float>()[index] = pixel.val[c];
			}
		}
	}
	input->copyFromHostTensor(inputHost.get());
}


static vector<PlateInfo> ProposalNet(unsigned char * inputImage, int height, int width, int minSize,
		float threshold, float factor) {

	float scale = 12.0f / minSize;
	float minWH = std::min(height, width) * scale;
	std::vector<float> scales;
	while (minWH >= minSize) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}
	total_boxes_.clear();

	for (int i = 0; i < scales.size(); i++) {
		int ws = (int) std::ceil(width * scales[i]);
		int hs = (int) std::ceil(height * scales[i]);
		std::vector<int> inputDims = { 1, 3, hs, ws };
		PNet_->resizeTensor(p_input, inputDims);

		PNet_->resizeSession(sess_p);

		MNN::CV::Matrix trans;
		trans.postScale(1.0f / ws, 1.0f / hs);
		trans.postScale(width, height);
		pretreat_data->setMatrix(trans);
		pretreat_data->convert(inputImage, width, height, 0, p_input);

		PNet_->runSession(sess_p);

		//onCopy to NCHW format
		Tensor prob_host(p_out_pro, p_out_pro->getDimensionType());
		Tensor reg_host(p_out_reg, p_out_reg->getDimensionType());
		p_out_pro->copyToHostTensor(&prob_host);
		p_out_reg->copyToHostTensor(&reg_host);
		auto * prob1_confidence = prob_host.host<float>() +  prob_host.stride(1);
		auto * reg = reg_host.host<float>();

		int feature_w = p_out_pro->width();
		int feature_h = p_out_pro->height();

		GenerateBBox(prob1_confidence, reg, feature_w, feature_h, scales[i], threshold);
		std::vector<PlateInfo> bboxes_nms = NMS(candidate_boxes_, 0.5f, 'u');
		if (bboxes_nms.size() > 0) {
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(),
					bboxes_nms.end());
		}
	}

	int num_box = (int) total_boxes_.size();
	vector<PlateInfo> res_boxes;
	if (num_box != 0) {
		res_boxes = NMS(total_boxes_, 0.7f, 'u');
		BBoxRegression(res_boxes);
		BBoxPadSquare(res_boxes, width, height);
	}
	return res_boxes;
}

/**
 * @sample is a normalized Mat
 */
static vector<PlateInfo> ProposalNet(const cv::Mat& sample, int minSize,
		float threshold, float factor) {
	int width = sample.cols;
	int height = sample.rows;
	float scale = factor;
	float minWH = std::min(height, width) * factor;
	std::vector<float> scales;
	while (minWH >= minSize) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}
	total_boxes_.clear();

	for (int i = 0; i < scales.size(); i++) {
		int ws = (int) std::ceil(width * scales[i]);
		int hs = (int) std::ceil(height * scales[i]);
		std::vector<int> inputDims = { 1, 3, hs,  ws };
		PNet_->resizeTensor(p_input, inputDims);
		PNet_->resizeSession(sess_p);
		fillInput(inputDims, sample, p_input);
		PNet_->runSession(sess_p);
//		MNN_PRINT("input tensor shape: %d %d %d %d\n", p_input->batch(), p_input->channel(), p_input->height(), p_input->width());

		Tensor prob_host(p_out_pro, p_out_pro->getDimensionType());
		Tensor reg_host(p_out_reg, p_out_reg->getDimensionType());
		p_out_pro->copyToHostTensor(&prob_host);
		p_out_reg->copyToHostTensor(&reg_host);
		auto * prob1_confidence = prob_host.host<float>() +  prob_host.stride(1);
		auto * reg = reg_host.host<float>();
		int feature_w = p_out_pro->width();
		int feature_h = p_out_pro->height();
//		MNN_PRINT("output tensor shape: %d %d %d %d \n", p_out_reg->batch(), p_out_reg->channel(),  p_out_reg->height(), p_out_reg->width());
		GenerateBBox(prob1_confidence, reg, feature_w, feature_h, scales[i], threshold);
		std::vector<PlateInfo> bboxes_nms = NMS(candidate_boxes_, 0.5f, 'u');
		if (bboxes_nms.size() > 0) {
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(),
					bboxes_nms.end());
		}
	}
	int num_box = (int) total_boxes_.size();
	vector<PlateInfo> res_boxes;
	if (num_box != 0) {
		res_boxes = NMS(total_boxes_, 0.7f, 'u');
		BBoxRegression(res_boxes);
		BBoxPadSquare(res_boxes, width, height);
	}
	return res_boxes;
}

static std::vector<PlateInfo> NextStage(const cv::Mat& sample,
		vector<PlateInfo> &pre_stage_res, int input_w, int input_h,
		int stage_num, const float threshold) {
	vector<PlateInfo> res;
	int batch_size = pre_stage_res.size();
	std::vector<int> inputDims = {1, 3, input_h, input_w };
	switch (stage_num) {
	case 2: {
		for (int n = 0; n < batch_size; ++n) {
			FaceBox & box = pre_stage_res[n].bbox;
			cv::Rect rect(cv::Point((int) box.xmin, (int) box.ymin),
					cv::Point((int) box.xmax, (int) box.ymax));
//			cout << "regression bbox: " << rect << endl;
			cv::Mat roi(sample, rect);
			imshow("rnet sample", roi);
			waitKey(0);
			fillInput(inputDims, roi, r_input);
			RNet_->runSession(sess_r);

			Tensor r_out_pro_host(r_out_pro, r_out_pro->getDimensionType());
			Tensor r_out_reg_host(r_out_reg, r_out_reg->getDimensionType());
			r_out_pro->copyToHostTensor(&r_out_pro_host);
			r_out_reg->copyToHostTensor(&r_out_reg_host);
			MNN_PRINT("rnet run session and on copy to host\n");
			auto confidence = r_out_pro_host.host<float>() + r_out_pro_host.stride(1);
			auto reg_box = r_out_reg_host.host<float>();

			float conf = confidence[0];
			if (conf >= threshold) {
				PlateInfo info;
				info.bbox.score = conf;
				info.bbox.xmin = pre_stage_res[n].bbox.xmin;
				info.bbox.ymin = pre_stage_res[n].bbox.ymin;
				info.bbox.xmax = pre_stage_res[n].bbox.xmax;
				info.bbox.ymax = pre_stage_res[n].bbox.ymax;
				for (int i = 0; i < 4; ++i) {
					info.bbox_reg[i] = reg_box[i];
				}
				res.push_back(info);
			}
		}
		break;
	}
	case 3: {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
		for (int n = 0; n < batch_size; ++n) {
			FaceBox &box = pre_stage_res[n].bbox;
			cv::Rect rect(cv::Point((int) box.xmin, (int) box.ymin),
					cv::Point((int) box.xmax, (int) box.ymax));
			cv::Mat roi(sample, rect);
			fillInput(inputDims, roi, o_input);
			ONet_->runSession(sess_o);

			Tensor o_out_pro_host(o_out_pro, o_out_pro->getDimensionType());
			Tensor o_out_reg_host(o_out_reg, o_out_reg->getDimensionType());
			Tensor o_out_lank_host(o_out_lank, o_out_lank->getDimensionType());
			o_out_pro->copyToHostTensor(&o_out_pro_host);
			o_out_reg->copyToHostTensor(&o_out_reg_host);
			o_out_lank->copyToHostTensor(&o_out_lank_host);

			auto confidence = o_out_pro_host.host<float>() + o_out_pro_host.stride(1);
			auto reg_box = o_out_reg_host.host<float>();
			auto reg_landmark = o_out_lank_host.host<float>();

			float conf = confidence[0];
			if (*confidence >= threshold) {
				PlateInfo info;
				info.bbox.score = conf;
				info.bbox.xmin = pre_stage_res[n].bbox.xmin;
				info.bbox.ymin = pre_stage_res[n].bbox.ymin;
				info.bbox.xmax = pre_stage_res[n].bbox.xmax;
				info.bbox.ymax = pre_stage_res[n].bbox.ymax;
				for (int i = 0; i < 4; ++i) {
					info.bbox_reg[i] = reg_box[i];
				}
				float w = info.bbox.xmax - info.bbox.xmin + 1.f;
				float h = info.bbox.ymax - info.bbox.ymin + 1.f;
				// x x x x y y y y to x y x y x y x y
				for (int i = 0; i < 4; ++i) {
					info.landmark[2 * i] = reg_landmark[2 * i] * w + info.bbox.xmin;
					info.landmark[2 * i + 1] = reg_landmark[2 * i + 1] * h + info.bbox.ymin;
				}
				res.push_back(info);
			}
		}
		break;
	}
	default:
		return res;
		break;
	}
	return res;
}

vector<PlateInfo> MLPDR::detect(unsigned char * inputImage, int height, int width, const int min_face, const int stage) {
	vector<PlateInfo> pnet_res;
	vector<PlateInfo> rnet_res;
	vector<PlateInfo> onet_res;

	if (stage >= 1) {
		pnet_res = ProposalNet(inputImage, height, width, min_face, threhold_p, factor);
	}

	if (stage == 1) {
		return pnet_res;
	} else if (stage == 2) {
		return rnet_res;
	} else if (stage == 3) {
		return onet_res;
	} else {
		return onet_res;
	}
}

std::vector<PlateInfo> MLPDR::recognize(const cv::Mat & img) {
	vector<PlateInfo> faceInfos = Detect(img, 40, 3);
	 recognize_plate_infos(img,  faceInfos);
	return faceInfos;
}
vector<PlateInfo> MLPDR::Detect(const cv::Mat & image, int min_face,
		int stage) {
	vector<PlateInfo> pnet_res;
	vector<PlateInfo> rnet_res;
	vector<PlateInfo> onet_res;

	cv::Mat sample = image.clone();
	sample.convertTo(sample, CV_32FC3, 0.0078125, -127.5*0.0078125);

	if (stage >= 1) {
		pnet_res = ProposalNet(sample, min_face, threhold_p, factor);
		MNN_PRINT("done ProposalNet with %d proposal \n", pnet_res.size());
	}

	if (stage >= 2 && pnet_res.size() > 0) {
		if (pnet_max_detect_num < (int) pnet_res.size()) {
			pnet_res.resize(pnet_max_detect_num);
		}
		rnet_res = NextStage(sample, pnet_res, 60, 24, 2, threhold_r);
		rnet_res = NMS(rnet_res, iou_threhold, 'u');
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, image.cols, image.rows);
	}
	if (stage >= 3 && rnet_res.size() > 0) {
		onet_res = NextStage(sample, rnet_res, 120, 48, 3, threhold_o);
		BBoxRegression(onet_res);
		onet_res = NMS(onet_res, iou_threhold, 'm');
		BBoxPad(onet_res, image.cols, image.rows);
	}
	if (stage == 1) {
		return pnet_res;
	} else if (stage == 2) {
		return rnet_res;
	} else if (stage == 3) {
		return onet_res;
	} else {
		return onet_res;
	}
}

static void extractMaxFace(vector<PlateInfo>& boundingBox_) {
	if (boundingBox_.empty()) {
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), CompareBBox);
	for (std::vector<PlateInfo>::iterator itx = boundingBox_.begin() + 1;
			itx != boundingBox_.end();) {
		itx = boundingBox_.erase(itx);
	}
}

std::vector<PlateInfo> MLPDR::Detect_MaxFace(const cv::Mat& img,
		const int min_face, const int stage) {
	vector<PlateInfo> pnet_res;
	vector<PlateInfo> rnet_res;
	vector<PlateInfo> onet_res;

	//total_boxes_.clear();
	//candidate_boxes_.clear();

	int width = img.cols;
	int height = img.rows;
	float scale = 12.0f / min_face;
	float minWH = std::min(height, width) * scale;
	std::vector<float> scales;
	while (minWH >= 24) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}

	std::reverse(scales.begin(), scales.end());

	uint8_t *pImg = get_img(img);
	for (int i = 0; i < scales.size(); i++) {
		int ws = (int) std::ceil(width * scales[i]);
		int hs = (int) std::ceil(height * scales[i]);
		std::vector<int> inputDims = { 1, 3, hs, ws };
		PNet_->resizeTensor(p_input, inputDims);
		PNet_->resizeSession(sess_p);

		MNN::CV::Matrix trans;
		trans.postScale(1.0f / ws, 1.0f / hs);
		trans.postScale(width, height);
		pretreat_data->setMatrix(trans);
		pretreat_data->convert(pImg, width, height, 0, p_input);

		PNet_->runSession(sess_p);
		float * confidence = p_out_pro->host<float>();
		float * reg = p_out_reg->host<float>();

		int feature_w = p_out_pro->width();
		int feature_h = p_out_pro->height();

		GenerateBBox(confidence, reg, feature_w, feature_h, scales[i],
				threhold_p);
		std::vector<PlateInfo> bboxes_nms = NMS(candidate_boxes_, 0.5f, 'u');

		//nmsTwoBoxs(bboxes_nms, pnet_res, 0.5);
		if (bboxes_nms.size() > 0) {
			pnet_res.insert(pnet_res.end(), bboxes_nms.begin(),
					bboxes_nms.end());
		} else {
			continue;
		}
		BBoxRegression(pnet_res);
		BBoxPadSquare(pnet_res, width, height);

		bboxes_nms.clear();
		bboxes_nms = NextStage(img, pnet_res, 60, 24, 2, threhold_r);
		bboxes_nms = NMS(bboxes_nms, iou_threhold, 'u');
		//nmsTwoBoxs(bboxes_nms, rnet_res, 0.5)
		if (bboxes_nms.size() > 0) {
			rnet_res.insert(rnet_res.end(), bboxes_nms.begin(),
					bboxes_nms.end());
		} else {
			pnet_res.clear();
			continue;
		}
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, img.cols, img.rows);

		onet_res = NextStage(img, rnet_res, 120, 48, 3, threhold_r);

		BBoxRegression(onet_res);
		onet_res = NMS(onet_res, iou_threhold, 'm');
		BBoxPad(onet_res, img.cols, img.rows);

		if (onet_res.size() < 1) {
			pnet_res.clear();
			rnet_res.clear();
			continue;
		} else {
			extractMaxFace(onet_res);
			delete pImg;
			return onet_res;
		}
	}
	delete pImg;
	return std::vector<PlateInfo> { };
}

} /* namespace mlpdr */
