#include <opencv/highgui.h>

#include <iostream>
#include <iomanip>
#ifndef FEATUREEXTRACTORS_H_
#define FEATUREEXTRACTORS_H_
using namespace cv;
namespace std {

class FeatureExtractors {
public:
	FeatureExtractors();
	virtual ~FeatureExtractors();
	double** cvCOL_Features(Mat,Mat);
};

} /* namespace std */

#endif /* FEATUREEXTRACTORS_H_ */
