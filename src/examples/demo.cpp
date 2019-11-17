/*
 * mtcnn_test.cpp
 *
 *  Created on: Jun 25, 2019
 *      Author: lqian
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <assert.h>
#include "mlpdr/MLPDR.h"

using namespace std;
using namespace mlpdr;
using namespace cv;
int main(int argc, char ** argv) {
	MLPDR detector(argv[1], 0.9f, 0.7f, 0.7f);
	Mat img = imread(argv[2]);
	assert(!img.empty());
	TickMeter tm;
	tm.start();
//	std::vector<mlpdr::PlateInfo> plateInfos = detector.Detect(img, 40, 3);  // 608.23 ms
	vector<PlateInfo> plateInfos = detector.recognize(img);
	tm.stop();
	printf("detect cost: %f (ms)\n", tm.getTimeMilli());

	for (auto pi: plateInfos) {
		cout << "plateNo: " << pi.plateNo << endl;
	}

}
