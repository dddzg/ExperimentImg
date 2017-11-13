#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;
namespace util {
/*双三阶插值的权重a*/
static float a=-0.5;
/*双三阶插值的x权重*/
void getW_x(float w_x[4], float x);
/*双三阶插值的y权重*/
void getW_y(float w_y[4], float y);
/*判断点坐标是否在矩阵中*/
bool within(Mat* mat, int x, int y);
}