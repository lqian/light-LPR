/*
 * mtcnn_test.cpp
 *
 *  Created on: Jun 25, 2019
 *      Author: lqian
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "mlpdr/MLPDR.h"

using namespace std;
using namespace mlpdr;
using namespace cv;
int main(int argc, char ** argv) {
	MLPDR detector(argv[1], 0.9f, 0.8f, 0.8f);
	Mat img = imread(argv[2]);
	int minFaceSize = 40;
	TickMeter tm;
	tm.start();
//	std::vector<mlpdr::FaceInfo> faceInfos = detector.Detect(img, minFaceSize, 3);  // 608.23 ms
	vector<string> plateNos;
	detector.recognizePlateNos(img, plateNos);
	tm.stop();
	printf("detect cost: %f (ms)\n", tm.getTimeMilli());

	for (auto plateNo: plateNos) {
		cout << "plateNo: " << plateNo << endl;
	}

}
