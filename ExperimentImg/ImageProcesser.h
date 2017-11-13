#pragma once
#include <iostream>
#include <cstring>
#include <string>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "util.h"
#define PI 3.14159265
using namespace std;
using namespace util;
// ÒýÓÃCUDA
extern "C" Mat* scaleUseCuda(Mat* mat,float n);
class ImageProcesser {
public:
	ImageProcesser(CImage* img,const CString& cstr,int threadNum=1,bool useGPU=false);
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
private:
	CImage* initImg;
	CImage* img;
	Mat* mat;
	CString cstr;
	int threadNum;
	bool useGPU;
};