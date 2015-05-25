#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "FeatureExtractors/hdr/FeatureExtractors.h"

using namespace std;
using namespace cv;

int main()
{
    int i;
    int j;
    FeatureExtractors extractFeatures;

    Mat image = imread("D:\\langage c\\ProjetSD\\WordSpotting\\images\\5744.jpg", CV_LOAD_IMAGE_COLOR);
    if(!image.data)
	{
		cout<<"Image loading problem!"<<endl;
		system("PAUSE");
	}

	Mat imGray;
	cvtColor(image,imGray,CV_BGR2GRAY);

	Mat imBinary = imGray > 128;

	//cout << imGray << endl;

	double** mRes;

    mRes = extractFeatures.cvCOL_Features(imGray, imBinary);
    unsigned int nCols = imBinary.cols;

    cout << imBinary.size() << endl;

//    for (i=0;i <nCols ;i++){
//        for (j=0;j<8;j++){
//            cout << mRes[i][j] << " ";
//        }
//        cout << endl;
//    }
//

//	namedWindow("Output", WINDOW_AUTOSIZE);
//	imshow("Output", imBinary);
//	imshow("Output", imGray);
//	waitKey(0);

	return 0;
}
