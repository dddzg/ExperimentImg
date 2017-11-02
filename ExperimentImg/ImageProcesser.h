#pragma once
#include <iostream>
#include <cstring>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <omp.h>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;
class ImageProcesser {
public:
	ImageProcesser(CImage* img,const CString& cstr,int threadNum=1,bool isCurrent=false);
	void MatToCImage(Mat& mat, CImage& cimage);
	void CImageToMat(CImage& cimage, Mat& mat);
	~ImageProcesser();
	CImage* go();
	Mat * salt(Mat* mat,int n);
	Mat * medianBlur(Mat* mat,int n);
private:
	CImage* initImg;
	CImage* img;
	Mat* mat;
	CString cstr;
	int threadNum;
	bool isCurrent;
};