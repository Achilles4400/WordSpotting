#include "hdr/FeatureExtractors.h"

namespace std {

FeatureExtractors::FeatureExtractors() {

}

FeatureExtractors::~FeatureExtractors() {
}

double** FeatureExtractors::cvCOL_Features(Mat matGrey_img,Mat matBin_img)
{
	unsigned int nRows = matGrey_img.rows;
	unsigned int nCols = matBin_img.cols;

	// Initialize Feature  matrix
	double **storFeatureMat = new double*[nCols]();
	for(unsigned int nXIndex=0; nXIndex<nCols; nXIndex++)
        storFeatureMat[nXIndex] = new double[9]();

    //used for feature 1
	unsigned int maxValue = 0;
	//used for feature 2
	unsigned int maxTransition = nRows/2;

	for (unsigned int getCol = 0;getCol<nCols;getCol++)
    {
        for (unsigned int getRwPixels = 0;getRwPixels<nRows;getRwPixels++)
        {
            if (matBin_img.at<unsigned int8_t>(getRwPixels,getCol) > maxValue)
                maxValue = matBin_img.at<unsigned int8_t>(getRwPixels,getCol);
        }
	}

	if ((!matGrey_img.empty()) && (!matBin_img.empty()))
        {
		for (unsigned int getCol = 0;getCol<nCols;getCol++)
            {
			//summation of the value of all pixel (for gray matrix), it will be used to calculate the projection profile (used for feature 1)
			unsigned int sumValueGrayPixel = 0;
			//store the position of the last bottom pixel found (used for feature 3 and 4)
			unsigned int upPixel = 0;
			unsigned int botPixel = 0;
			//store the number of background to ink pixels (used for feature 2)
			unsigned int backToInkPixel = 0;
			unsigned int lastPixel = 255;
			//flag = 0 until no black pixel found (used for feature 3)
			int	pixelFlag = 0;
			//count the number of black pixel (used for feature 5)
			unsigned int nbForegroundPixel = 0;
			//summation of black pixels position (used for feature 6)
			unsigned int sumForegroundPixel = 0;

			for (unsigned int getRwPixels = 0;getRwPixels<nRows;getRwPixels++)
            {   // starting from minimum row to maximum row, sum all pixels in between them
                //the type of values inside the matrix is unsigned int8_t (0 to 255 value)
                // gray matrix part
                sumValueGrayPixel += (255 - matGrey_img.at<unsigned int8_t>(getRwPixels,getCol));
                // binary matrix part
                //this is the test to store the first foreground pixel and it will also used to store the last foreground pixel
                    if(matBin_img.at<unsigned int8_t>(getRwPixels,getCol) == 0)
                    { // in opencv, back ground pixels are 255 (blanc) and fore ground pixels are 0 (noir)
                        if(pixelFlag == 1)
                        {
                            //setting up the lower profile (feature 4) the last foreground is stored DIV par nbLignes
                            botPixel = getRwPixels;
                            if (lastPixel == 255)
                            {
                                backToInkPixel++;
                            }
                            nbForegroundPixel++;
                            sumForegroundPixel += getRwPixels;
                        }
                        if(pixelFlag == 0)
                        {
                            //setting up the upper profile (feature 3) the first foreground pixel is stored
                            upPixel = getRwPixels;
                            storFeatureMat[getCol][2] = (double)upPixel/nRows;
                            //store the last bot pixel
                            botPixel = getRwPixels;
                            backToInkPixel++;
                            nbForegroundPixel++;
                            sumForegroundPixel += getRwPixels;
                            pixelFlag = 1;
                        }
                    }
				//setting the last pixel
				lastPixel = matBin_img.at<unsigned int8_t>(getRwPixels,getCol);
				}
            // need to cast feature in double because all values are unsigned int
            // Binary level features
            storFeatureMat[getCol][0] = (double)sumValueGrayPixel/maxValue; // projection profile (feature 1)
            storFeatureMat[getCol][1] = (double)backToInkPixel/maxTransition; // background to ink transition (feature 2)
            if (pixelFlag == 0)
            {//if no pixel found setting feature 3 and 4 at -1
                storFeatureMat[getCol][2] = -1;
                storFeatureMat[getCol][3] = -1;
            }
            else
                storFeatureMat[getCol][3] = (double)botPixel/nRows; // bottom pixel (feature 4)
            storFeatureMat[getCol][4] = (double)storFeatureMat[getCol][3] - (double)storFeatureMat[getCol][2]; // distance between upper and lower profile (feature 5)
            storFeatureMat[getCol][5] = nbForegroundPixel; // number of foreground pixels (feature 6)
            if (nbForegroundPixel != 0)
                storFeatureMat[getCol][6] = sumForegroundPixel/nbForegroundPixel; // gravity center of the column (feature 7)
            else//if no black pixel in the column
                storFeatureMat[getCol][6] = -1;

            //calculation of feature 8
            //if gravity center equal -1 (white column)
            if (storFeatureMat[getCol][6] == -1)
            {
                storFeatureMat[getCol][7] = 0;
            }
            //if at gravity center index and gravity center index +/-1 equal black white or white black then feature 8 equal 1
            else if ((matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6],getCol) == 0 && matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6]-1,getCol) == 255) ||
                    (matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6],getCol) == 0 && matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6]+1,getCol) == 255) ||
                    (matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6],getCol) == 255 && matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6]-1,getCol) == 0) ||
                    (matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6],getCol) == 255 && matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6]+1,getCol) == 0))
            {
                storFeatureMat[getCol][7] = 1;
            }
            else
            {
                storFeatureMat[getCol][7] = 0;
            }

            //calculation of feature 9
            if (getCol != 0)
            {
                //if the gravity center for the tow columns exist
                if (storFeatureMat[getCol][6] != -1 && storFeatureMat[getCol-1][6] !=-1)
                    storFeatureMat[getCol][8] = storFeatureMat[getCol][6] - storFeatureMat[getCol-1][6];
                else //if one of gravity center equal -1 (white column) TODO réfléchir sur la valeur a retourner
                    storFeatureMat[getCol][8] = 0;
            }
            else
            {
                //feature 9 is equal to 0 for first column
                storFeatureMat[getCol][8] = 0;
            }

            //calculation of, feature 8 the value is 1 if there is a transition 0 otherwise
            //remplacé par F8, F9
//            if(getCol > 0)
//            {
//                if((matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6],getCol) ==255 && matBin_img.at<unsigned int8_t>(storFeatureMat[getCol-1][6],getCol) == 0)
//                   || (matBin_img.at<unsigned int8_t>(storFeatureMat[getCol][6],getCol) ==0 && matBin_img.at<unsigned int8_t>(storFeatureMat[getCol-1][6],getCol) == 255))
//                {
//                    storFeatureMat[getCol][7] = 1; // if there is a transition between column ans column-1
//                }
//                else
//                {
//                    storFeatureMat[getCol][7] = 0; // if there is no transition
//                }
//            }
//            else
//            {
//                //default value for the column 0, do not pay attention for it
//                storFeatureMat[getCol][7] = 0;
//            }

            cout << fixed << setprecision(2) << storFeatureMat[getCol][0]
            << " || " << storFeatureMat[getCol][1] << " || "
            << storFeatureMat[getCol][2] << " || " << storFeatureMat[getCol][3] << " || " << storFeatureMat[getCol][4] << " || "
            << storFeatureMat[getCol][5] << " || " << storFeatureMat[getCol][6] << " || " << storFeatureMat[getCol][7] << " || "
            << storFeatureMat[getCol][8] <<endl;
		}
		return storFeatureMat ;  //refinedStorFearureMat
	}
	else{
		return NULL;  // throw an error saying the image is empty
	}
}
}
