/*
 * api.cpp
 *
 *  Created on: Jul 14, 2019
 *      Author: link
 */


#include <cstdlib>
#include <string>
#include <memory>
#include <string.h>
#include <vector>
#include <iostream>

#include <mlpdr/MLPDR.h>

using namespace std;
using namespace mlpdr;
using namespace cv;

mlpdr::MLPDR * ptr_mlpdr;

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined CVAPI_EXPORTS
#  define API_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define API_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define API_EXPORTS
#endif

#ifdef __cplusplus
extern "C" {
#endif


API_EXPORTS void coreInitContext() {
	ptr_mlpdr = new MLPDR("/home/lqian/cpp-workspace/light-LPDR/models/");
}

API_EXPORTS void cleanContext() {
	delete ptr_mlpdr;
}

/**
 * @res resource URL represent a image to be recognized
 * @contentType char buffer, full filename, network resource etc
 * @output a char point that point output content buffer
 */
API_EXPORTS int recogSingleJson(char * res, int contentType, char ** output) {
	Mat img = imread(res);
	if (img.empty()) {
		return -1;
	}

	vector<PlateInfo> plateInfos = ptr_mlpdr->recognize(img);
	if (plateInfos.size() == 0) return 0;
	string plateNo = plateInfos[0].plateNo;  //only return 0 for highest confidence plate
	int len = plateNo.length();
	*output = (char *) calloc(len, sizeof(char));
	memcpy(*output, plateNo.c_str(), len);
	return len;
}

#ifdef __cplusplus
}
#endif

