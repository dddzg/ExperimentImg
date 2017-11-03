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
#include <cmath>
#define PI 3.14159265
using namespace cv;
using namespace std;
class ImageProcesser {
public:
	ImageProcesser(CImage* img,const CString& cstr,int threadNum=1,bool isCurrent=false);
	static void MatToCImage(Mat& mat, CImage& cimage);
	static void CImageToMat(CImage& cimage, Mat& mat);
	~ImageProcesser();
	CImage* go();
	Mat * salt(Mat* mat,int n);
	Mat * medianBlur(Mat* mat,int n);
	Mat * scale(Mat* mat, float n);
	Mat * rotate(Mat* mat, float angle);
	Mat * autoBalance(Mat* mat);
	Mat * autoLevel(Mat* mat);
	Mat * bilateralFilter(Mat * mat, int d, double sigmaColor, double sigmaSpace );
	static CImage* merge(CImage* src, CImage* dist, double alpha);
	static float a;
private:
	CImage* initImg;
	CImage* img;
	Mat* mat;
	CString cstr;
	int threadNum;
	bool isCurrent;
	void getW_x(float w_x[4], float x);
	void getW_y(float w_y[4], float y);
};