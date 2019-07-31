/*
 * LPRDetector.h
 *
 *  Created on: Jun 25, 2019
 *      Author: lqian
 */

#ifndef INCLUDE_MLPDR_MLPDR_H_
#define INCLUDE_MLPDR_MLPDR_H_

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using std::string;
using std::vector;

namespace mlpdr {

typedef struct FaceBox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
} FaceBox;
typedef struct FaceInfo {
	float bbox_reg[4];
	float landmark_reg[8];
	float landmark[8];
	FaceBox bbox;
} FaceInfo;

class MLPDR {
public:
	MLPDR(const string& proto_model_dir, float threhold_p = 0.7f,
			float threhold_r = 0.8f, float threhold_o = 0.8f, float factor =
					0.709f);
	void recognizePlateNos(const cv::Mat & img, std::vector<std::string> & plateNos);
	std::vector<std::vector<int>> recognize(const cv::Mat & img);
	std::vector<FaceInfo> Detect(const cv::Mat& img, const int min_face = 64,
			const int stage = 3);
	vector<FaceInfo> detect(unsigned char * inputImage, int height, int width, const int min_face=64, const int stagee=3);
	std::vector<FaceInfo> Detect_MaxFace(const cv::Mat& img,
			const int min_face = 64, const int stage = 3);

	virtual ~MLPDR();
private:
	int threads_num = 2;
};

} /* namespace mlpdr */

#endif /* INCLUDE_MLPDR_MLPDR_H_ */
