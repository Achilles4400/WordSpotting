/*
 * FeatureExtractors.cpp
 *
 *  Created on: Feb 7, 2015
 *      Author: tanmoymondal
 */

#include "hdr/FeatureExtractors.h"
#include <math.h>
//#include "mex.h"
#define round(x) (x<0?ceil((x)-0.5):floor((x)+0.5))
// small value, used to avoid division by zero
#define eps 0.0001

namespace std {

FeatureExtractors::FeatureExtractors() {

}

FeatureExtractors::~FeatureExtractors() {
}
Ptr<DescriptorExtractor> FeatureExtractors::cvSIFT_Features(IplImage* src_img){
	Mat image(src_img);
	// Create smart pointer for SIFT feature detector.
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	vector<KeyPoint> keypoints;

	// Detect the key points
	featureDetector->detect(image, keypoints); // NOTE: featureDetector is a pointer hence the '->'.

	//Similarly, we create a smart pointer to the SIFT extractor.
	Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");

	// Compute the 128 dimension SIFT descriptor at each key point.
	// Each row in "descriptors" correspond to the SIFT descriptor for each key point
	Mat descriptors;
	featureExtractor->compute(image, keypoints, descriptors);

	// If you would like to draw the detected key point just to check
//
//    Mat outputImage;
//    Scalar keypointColor = Scalar(255, 0, 0);     // Blue key points.
//    drawKeypoints(image, keypoints, outputImage, keypointColor, DrawMatchesFlags::DEFAULT);
//
//    namedWindow("Output");
//    imshow("Output", outputImage);
	return featureExtractor;
}

double** FeatureExtractors::cvCOL_Features(Mat matGrey_img,Mat matBin_img){
	unsigned int nRows = matGrey_img.rows;
	unsigned int nCols = matBin_img.cols;

	// Initialize Feature  matrix
	double **storFeatureMat = new double*[nCols]();
    for(unsigned int nXIndex=0; nXIndex<nCols; nXIndex++)
        storFeatureMat[nXIndex] = new double[8]();

	// Initialize topIndex  matrix
	int storTopIndex[nCols][3];

	// Initialize botIndex  matrix
	int storBottomIndex[nCols][3];

	// Initialize storForeGroundCGforCol  matrix
	int storForeGroundCGforCol[nCols][3];

	if ((!matGrey_img.empty()) && (!matBin_img.empty())){
		unsigned int foreGroundColCnt = 0;
		for (unsigned int getCol = 0;getCol<nCols;getCol++){
			int	calTransition = 0;
			unsigned int nEdgePixels = 0;
			int botPixel = 0;
			int	pixelFlag = 0;
			int	sumEdgePixels[nRows];
			int storRwColOfEdgePixels[nRows][2];

            memset(sumEdgePixels, 0, sizeof(sumEdgePixels)); // fr initializing the array by 0
			memset(storRwColOfEdgePixels, 0, sizeof(storRwColOfEdgePixels[0][0]) * nRows * 2);

			for (unsigned int getRwPixels = 0;getRwPixels<nRows;getRwPixels++){ // starting from minimum row to maximum row, sum all pixels in between them

                //this is the test to store the first foreground pixel and it will also used to store the last foreground pixel
                //the type of values inside the matrix is unsigned int8_t
				if(matBin_img.at<unsigned int8_t>(getRwPixels,getCol) == 255){ // in opencv, fore ground pixels are 255 and back ground pixels are 0
					if(pixelFlag == 0){
					    //setting up the upper profile (feature 3) the first foreground pixel is stored

					    storFeatureMat[getCol][2] = getRwPixels;
					    //store the last bot pixel
					    botPixel = getRwPixels;
//
//						storTopIndex[getCol][0] = getRwPixels; // storing the row
//						storTopIndex[getCol][1] = getCol; // storing the col
//						storTopIndex[getCol][1] = sqrt((getRwPixels-0)^2); // storing the col
						pixelFlag = 1;
					}
					else if(pixelFlag == 1){
					    //setting up the lower profile (feature 4) the last foreground is stored
					    botPixel = getRwPixels;
//
//						storBottomIndex[getCol][0] = getRwPixels; // storing the row
//						storBottomIndex[getCol][1] = getCol; // storing the col
//						storBottomIndex[getCol][1] = sqrt((getRwPixels-nRows)^2); // storing the col
					}
					sumEdgePixels[getRwPixels] =
							(255 - (matGrey_img.at<unsigned int8_t>(getRwPixels,getCol) )); // for edge pixels only, store the grey values
					nEdgePixels = nEdgePixels + 1;
					storRwColOfEdgePixels[nEdgePixels][0] = getRwPixels;
					storRwColOfEdgePixels[nEdgePixels][1] = getCol;
				}
				if(getRwPixels> 0){
					if (  (matBin_img.at<unsigned int8_t>(getRwPixels,getCol)== 255) &&
							(matBin_img.at<unsigned int8_t>(getRwPixels-1,getCol) == 0)  ){// AS WE WANT ONLY BACK GROUND TO INK TRANSITION
						calTransition = calTransition+1;
					}
				}
			}

//			if(nEdgePixels == 1) {// If only one pixel exists in the column, then bottom pixel will also be same as top pixel
//				storBottomIndex[getCol][0] = storTopIndex[getCol][0];
//				storBottomIndex[getCol][1] = storTopIndex[getCol][1];
//				storBottomIndex[getCol][2] = sqrt((storTopIndex[getCol][0]-nRows)^2);
//			}
            int myArrSum = 0;
            for (unsigned int calcSum = 0;calcSum<nEdgePixels;calcSum++)
                myArrSum = myArrSum + storRwColOfEdgePixels[calcSum][0];
            int cgOfRw = myArrSum/nEdgePixels;
            int cgOfCol = storRwColOfEdgePixels[0][1];
            foreGroundColCnt = foreGroundColCnt +1;

            storForeGroundCGforCol[foreGroundColCnt][0] = round(cgOfRw);
            storForeGroundCGforCol[foreGroundColCnt][1] = round(cgOfCol);
            storForeGroundCGforCol[foreGroundColCnt][2] = getCol;
            // Binary level features
            myArrSum = 0;
            for (unsigned int calcSum = 0;calcSum<nRows;calcSum++)
                myArrSum = myArrSum + sumEdgePixels[calcSum];
            storFeatureMat[getCol][0] = myArrSum/nRows;
            storFeatureMat[getCol][1] = nEdgePixels/nRows; // projection profile
            //storFeatureMat[getCol][2] = storTopIndex[getCol][2]/nRows;// storing the upper profile; as we are calculating the
//				storFeatureMat[getCol][3] = storBottomIndex[getCol][2]/nRows; // storing the lower profile;as we are calculating the
            storFeatureMat[getCol][3] = botPixel;
            storFeatureMat[getCol][4] = ( storBottomIndex[getCol][2] - storTopIndex[getCol][2] )/nRows;
            storFeatureMat[getCol][5] = calTransition / 10;
            storFeatureMat[getCol][6] = cgOfRw/nRows;

            cout << storFeatureMat[getCol][2] << " " << botPixel << endl;


            //show feature 0 to 6 in the output, to check if values are OK
//				cout << myArrSum/nRows << " " << nEdgePixels/nRows << " " << storFeatureMat[getCol][2]/nRows << " " << storBottomIndex[getCol][2]/nRows << " ";
//				cout << ( storBottomIndex[getCol][2] - storTopIndex[getCol][2] )/nRows << " " << calTransition / 10 << " " << cgOfRw/nRows;
//				cout << endl;
		}
		for (unsigned int ii = 0;ii<foreGroundColCnt;ii++){
			if(ii > 0){
				if(  ((matBin_img.at<unsigned int8_t>(  (storForeGroundCGforCol[ii][0]),(storForeGroundCGforCol[ii][1])  ) == 0) &&
						(matBin_img.at<unsigned int8_t>(  (storForeGroundCGforCol[ii-1][0]),(storForeGroundCGforCol[ii-1][1]) == 255)) )  ||
						( (matBin_img.at<unsigned int8_t>( (storForeGroundCGforCol[ii][0]),(storForeGroundCGforCol[ii][1]) ) == 255) &&
								(matBin_img.at<unsigned int8_t>( (storForeGroundCGforCol[ii-1][0]),(storForeGroundCGforCol[ii-1][1]) ) == 0) ) ){
					storFeatureMat[(storForeGroundCGforCol[ii][1])][7] = 1; // storing the transition of the CG of each foreground pixels in the column
				}
				else{
					storFeatureMat[(storForeGroundCGforCol[ii][1])][7] = 0; // if there is no transition but this column have foreground pixel
				}
			}
		}

		// For spline interpolation; get the indexes where storFeatureMat is non zero
//		std::vector<int> non0Rw,x_inter;
//		for (unsigned int t = 0; t<nCols;t++){
//			if(storFeatureMat[t][0] != 0)
//				non0Rw.push_back(t);
//		}
//		x_inter = non0Rw;
//		set<double> x_inter_set;
//		copy(x_inter.begin(), x_inter.end(), inserter(x_inter_set, x_inter_set.begin()));
//		for (unsigned int pii = 0; pii<8; pii++){ // for all the features, as there are 8 features so it is 8
//			set<double> y_inter_set,x_req;
//			std::vector<double> Yinter;
//			for (unsigned int qii = 0; qii<8; qii++){
//				double val = storFeatureMat[qii][pii];
//				Yinter.push_back(val);
//			}
//			copy(Yinter.begin(), Yinter.end(), inserter(y_inter_set, y_inter_set.begin()));
//			std::set_difference(x_inter_set.begin(), x_inter_set.end(), y_inter_set.begin(),
//					y_inter_set.end(),std::inserter(x_req, x_req.end()));
//			// converting set to vector
//			std::vector<int> output_intersection_vect;
//			std::copy(x_req.begin(), x_req.end(), inserter(output_intersection_vect, output_intersection_vect.begin()));
//			//copy vector to mat
//			Mat x_inter_mat,y_inter_mat;
//			memcpy(x_inter_mat.data,x_inter.data(),x_inter.size()*sizeof(int));
//			memcpy(y_inter_mat.data,Yinter.data(),Yinter.size()*sizeof(double));
//			for (unsigned int gap_fil = 0; gap_fil<output_intersection_vect.size(); gap_fil++){
//				double missing_val = 0;//interpolatethis->interpolate(x_inter_mat, y_inter_mat,int(output_intersection_vect.at(gap_fil)) );
//				storFeatureMat[output_intersection_vect.at(gap_fil)][pii] = missing_val;
//			}
//
//		}
		// Initialize Feature  matrix
//		double refinedStorFearureMat[nCols][8];

//		int	lukUpTableForRealIndex[nRows];
//		memset(lukUpTableForRealIndex, 0, sizeof(lukUpTableForRealIndex));

//		for (unsigned int h = 0;h<nCols;h++){
//			for (unsigned int g = 0;g<8;g++){
//				refinedStorFearureMat[h][g] = storFeatureMat[h][g];
//				double val = refinedStorFearureMat[h][g];
//				if(isnan(val)){
//					refinedStorFearureMat[h][g] = 0;
//				}
//			}
//			lukUpTableForRealIndex[h] = h;
//		}
		return storFeatureMat ;  //refinedStorFearureMat
	}
	else{
		return NULL;  // throw an error saying the image is empty
	}

}
double interpolate(int x1, double y1, int x2, double y2, int targetX)
{
	int diffX = x2 - x1;
	double diffY = y2 - y1;
	int diffTarget = targetX - x1;

	return y1 + (diffTarget * diffY) / diffX;
}
double interpolate(Mat X, Mat Y, int targetX)
{
	Mat dist = abs(X-targetX);
	double minVal, maxVal;
	Point minLoc1, minLoc2, maxLoc;

	// find the nearest neighbour
	Mat mask = Mat::ones(X.rows, X.cols, CV_8UC1);
	minMaxLoc(dist,&minVal, &maxVal, &minLoc1, &maxLoc, mask);

	// mask out the nearest neighbour and search for the second nearest neighbour
	mask.at<uchar>(minLoc1) = 0;
	minMaxLoc(dist,&minVal, &maxVal, &minLoc2, &maxLoc, mask);

	// use the two nearest neighbours to interpolate the target value
	double res = interpolate(X.at<int>(minLoc1), Y.at<double>(minLoc1), X.at<int>(minLoc2), Y.at<double>(minLoc2), targetX);
	return res;
}

//vector<float> FeatureExtractors::cvHOG_Features(IplImage* src_img){
//// unit vectors used to compute gradient orientation
//double uu[9] = {1.0000,
//		0.9397,
//		0.7660,
//		0.500,
//		0.1736,
//		-0.1736,
//		-0.5000,
//		-0.7660,
//		-0.9397};
//double vv[9] = {0.0000,
//		0.3420,
//		0.6428,
//		0.8660,
//		0.9848,
//		0.9848,
//		0.8660,
//		0.6428,
//		0.3420};
//
//static inline double min(double x, double y) { return (x <= y ? x : y); }
//static inline double max(double x, double y) { return (x <= y ? y : x); }
//
//static inline int min(int x, int y) { return (x <= y ? x : y); }
//static inline int max(int x, int y) { return (x <= y ? y : x); }
//
//// main function:
//// takes a double color image and a bin size
//// returns HOG features
//mxArray *process(const mxArray *mximage, const mxArray *mxsbin) {
//  double *im = (double *)mxGetPr(mximage);
//  const int *dims = mxGetDimensions(mximage);
//  if (mxGetNumberOfDimensions(mximage) != 3 ||
//      dims[2] != 3 ||
//      mxGetClassID(mximage) != mxDOUBLE_CLASS)
//    mexErrMsgTxt("Invalid input");
//
//  int sbin = (int)mxGetScalar(mxsbin);
//
//  // memory for caching orientation histograms & their norms
//  int blocks[2];
//  blocks[0] = (int)round((double)dims[0]/(double)sbin);
//  blocks[1] = (int)round((double)dims[1]/(double)sbin);
//  double *hist = (double *)mxCalloc(blocks[0]*blocks[1]*18, sizeof(double));
//  double *norm = (double *)mxCalloc(blocks[0]*blocks[1], sizeof(double));
//
//  // memory for HOG features
//  int out[3];
//  out[0] = max(blocks[0]-2, 0);
//  out[1] = max(blocks[1]-2, 0);
//  out[2] = 27+4;
//  mxArray *mxfeat = mxCreateNumericArray(3, out, mxDOUBLE_CLASS, mxREAL);
//  double *feat = (double *)mxGetPr(mxfeat);
//
//  int visible[2];
//  visible[0] = blocks[0]*sbin;
//  visible[1] = blocks[1]*sbin;
//
//  for (int x = 1; x < visible[1]-1; x++) {
//    for (int y = 1; y < visible[0]-1; y++) {
//      // first color channel
//      double *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
//      double dy = *(s+1) - *(s-1);
//      double dx = *(s+dims[0]) - *(s-dims[0]);
//      double v = dx*dx + dy*dy;
//
//      // second color channel
//      s += dims[0]*dims[1];
//      double dy2 = *(s+1) - *(s-1);
//      double dx2 = *(s+dims[0]) - *(s-dims[0]);
//      double v2 = dx2*dx2 + dy2*dy2;
//
//      // third color channel
//      s += dims[0]*dims[1];
//      double dy3 = *(s+1) - *(s-1);
//      double dx3 = *(s+dims[0]) - *(s-dims[0]);
//      double v3 = dx3*dx3 + dy3*dy3;
//
//      // pick channel with strongest gradient
//      if (v2 > v) {
//	v = v2;
//	dx = dx2;
//	dy = dy2;
//      }
//      if (v3 > v) {
//	v = v3;
//	dx = dx3;
//	dy = dy3;
//      }
//
//      // snap to one of 18 orientations
//      double best_dot = 0;
//      int best_o = 0;
//      for (int o = 0; o < 9; o++) {
//	double dot = uu[o]*dx + vv[o]*dy;
//	if (dot > best_dot) {
//	  best_dot = dot;
//	  best_o = o;
//	} else if (-dot > best_dot) {
//	  best_dot = -dot;
//	  best_o = o+9;
//	}
//      }
//
//      // add to 4 histograms around pixel using linear interpolation
//      double xp = ((double)x+0.5)/(double)sbin - 0.5;
//      double yp = ((double)y+0.5)/(double)sbin - 0.5;
//      int ixp = (int)floor(xp);
//      int iyp = (int)floor(yp);
//      double vx0 = xp-ixp;
//      double vy0 = yp-iyp;
//      double vx1 = 1.0-vx0;
//      double vy1 = 1.0-vy0;
//      v = sqrt(v);
//
//      if (ixp >= 0 && iyp >= 0) {
//	*(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
//	  vx1*vy1*v;
//      }
//
//      if (ixp+1 < blocks[1] && iyp >= 0) {
//	*(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
//	  vx0*vy1*v;
//      }
//
//      if (ixp >= 0 && iyp+1 < blocks[0]) {
//	*(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
//	  vx1*vy0*v;
//      }
//
//      if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
//	*(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
//	  vx0*vy0*v;
//      }
//    }
//  }
//
//  // compute energy in each block by summing over orientations
//  for (int o = 0; o < 9; o++) {
//    double *src1 = hist + o*blocks[0]*blocks[1];
//    double *src2 = hist + (o+9)*blocks[0]*blocks[1];
//    double *dst = norm;
//    double *end = norm + blocks[1]*blocks[0];
//    while (dst < end) {
//      *(dst++) += (*src1 + *src2) * (*src1 + *src2);
//      src1++;
//      src2++;
//    }
//  }
//
//  // compute features
//  for (int x = 0; x < out[1]; x++) {
//    for (int y = 0; y < out[0]; y++) {
//      double *dst = feat + x*out[0] + y;
//      double *src, *p, n1, n2, n3, n4;
//
//      p = norm + (x+1)*blocks[0] + y+1;
//      n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//      p = norm + (x+1)*blocks[0] + y;
//      n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//      p = norm + x*blocks[0] + y+1;
//      n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//      p = norm + x*blocks[0] + y;
//      n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//
//      double t1 = 0;
//      double t2 = 0;
//      double t3 = 0;
//      double t4 = 0;
//
//      // contrast-sensitive features
//      src = hist + (x+1)*blocks[0] + (y+1);
//      for (int o = 0; o < 18; o++) {
//	double h1 = min(*src * n1, 0.2);
//	double h2 = min(*src * n2, 0.2);
//	double h3 = min(*src * n3, 0.2);
//	double h4 = min(*src * n4, 0.2);
//	*dst = 0.5 * (h1 + h2 + h3 + h4);
//	t1 += h1;
//	t2 += h2;
//	t3 += h3;
//	t4 += h4;
//	dst += out[0]*out[1];
//	src += blocks[0]*blocks[1];
//      }
//
//      // contrast-insensitive features
//      src = hist + (x+1)*blocks[0] + (y+1);
//      for (int o = 0; o < 9; o++) {
//        double sum = *src + *(src + 9*blocks[0]*blocks[1]);
//        double h1 = min(sum * n1, 0.2);
//        double h2 = min(sum * n2, 0.2);
//        double h3 = min(sum * n3, 0.2);
//        double h4 = min(sum * n4, 0.2);
//        *dst = 0.5 * (h1 + h2 + h3 + h4);
//        dst += out[0]*out[1];
//        src += blocks[0]*blocks[1];
//      }
//
//      // texture features
//      *dst = 0.2357 * t1;
//      dst += out[0]*out[1];
//      *dst = 0.2357 * t2;
//      dst += out[0]*out[1];
//      *dst = 0.2357 * t3;
//      dst += out[0]*out[1];
//      *dst = 0.2357 * t4;
//    }
//  }
//
//  mxFree(hist);
//  mxFree(norm);
//  return mxfeat;
//}

// matlab entry point
// F = features_pedro(image, bin)
// image should be color with double values
//void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
//  if (nrhs != 2)
//    mexErrMsgTxt("Wrong number of inputs");
//  if (nlhs != 1)
//    mexErrMsgTxt("Wrong number of outputs");
//  plhs[0] = process(prhs[0], prhs[1]);
//}






vector<float> FeatureExtractors::cvHOG_Features(IplImage* src_img){
	HOGDescriptor hog; // Use standard parameters here
	static const Size trainingPadding = Size(0, 0);
	static const Size winStride = Size(8, 8);
	hog.winSize = Size(64, 128); // Default training images size as used in paper
	vector<float> featureVector;
	Mat imageData(src_img );
	if (imageData.empty()) {
		featureVector.clear();
		std::cerr << "Error: HOG image is empty, features calculation skipped!\n";
		std::terminate();
	}
	// Check for mismatching dimensions
	if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
		featureVector.clear();
		printf("Error: Image dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
		std::cerr << "Error: See the previous line for error";
		std::terminate();
	}
	vector<Point> locations;
	hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
	imageData.release(); // Release the image again after features are extracted
	return featureVector;
}
}
