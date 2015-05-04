/*
 * FeatureExtractors.h
 *
 *  Created on: Feb 7, 2015
 *      Author: tanmoymondal
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <set>
#include <iterator>
#ifndef FEATUREEXTRACTORS_H_
#define FEATUREEXTRACTORS_H_
using namespace cv;
namespace std {

class FeatureExtractors {
public:
	FeatureExtractors();
	virtual ~FeatureExtractors();
	Ptr<DescriptorExtractor> cvSIFT_Features(IplImage*);
	double interpolate(int, double, int, double, int);
	double interpolate(Mat, Mat, int);
	double** cvCOL_Features(Mat,Mat);
	//Mat cvSURF_Features(IplImage*);
	vector<float> cvHOG_Features(IplImage*);
};

} /* namespace std */

#endif /* FEATUREEXTRACTORS_H_ */
