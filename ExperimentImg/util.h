#pragma once
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;
namespace util {
/*˫���ײ�ֵ��Ȩ��a*/
static float a=-0.5;
/*˫���ײ�ֵ��xȨ��*/
void getW_x(float w_x[4], float x);
/*˫���ײ�ֵ��yȨ��*/
void getW_y(float w_y[4], float y);
/*�жϵ������Ƿ��ھ�����*/
bool within(Mat* mat, int x, int y);
}