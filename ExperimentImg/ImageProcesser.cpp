#include "stdafx.h"
#include "ImageProcesser.h"



ImageProcesser::ImageProcesser(CImage * img, const CString & cstr, int threadNum, bool isCurrent)
{
	this->initImg = img;
	this->mat = new Mat();
	this->cstr = move(cstr);
	this->threadNum = threadNum;
	this->isCurrent = isCurrent;
	this->CImageToMat(*this->initImg, *this->mat);
}

void ImageProcesser::MatToCImage(Mat & mat, CImage & cimage)
{
	if (0 == mat.total())
	{
		return;
	}


	int nChannels = mat.channels();
	if ((1 != nChannels) && (3 != nChannels))
	{
		return;
	}
	int nWidth = mat.cols;
	int nHeight = mat.rows;


	//�ؽ�cimage  
	cimage.Destroy();
	cimage.Create(nWidth, nHeight, 8 * nChannels);


	//��������  


	uchar* pucRow;                                  //ָ������������ָ��  
	uchar* pucImage = (uchar*)cimage.GetBits();     //ָ����������ָ��  
	int nStep = cimage.GetPitch();                  //ÿ�е��ֽ���,ע���������ֵ�����и�  


	if (1 == nChannels)                             //���ڵ�ͨ����ͼ����Ҫ��ʼ����ɫ��  
	{
		RGBQUAD* rgbquadColorTable;
		int nMaxColors = 256;
		rgbquadColorTable = new RGBQUAD[nMaxColors];
		cimage.GetColorTable(0, nMaxColors, rgbquadColorTable);
		for (int nColor = 0; nColor < nMaxColors; nColor++)
		{
			rgbquadColorTable[nColor].rgbBlue = (uchar)nColor;
			rgbquadColorTable[nColor].rgbGreen = (uchar)nColor;
			rgbquadColorTable[nColor].rgbRed = (uchar)nColor;
		}
		cimage.SetColorTable(0, nMaxColors, rgbquadColorTable);
		delete[]rgbquadColorTable;
	}


	for (int nRow = 0; nRow < nHeight; nRow++)
	{
		pucRow = (mat.ptr<uchar>(nRow));
		for (int nCol = 0; nCol < nWidth; nCol++)
		{
			if (1 == nChannels)
			{
				*(pucImage + nRow * nStep + nCol) = pucRow[nCol];
			}
			else if (3 == nChannels)
			{
				for (int nCha = 0; nCha < 3; nCha++)
				{
					*(pucImage + nRow * nStep + nCol * 3 + nCha) = pucRow[nCol * 3 + nCha];
				}
			}
		}
	}
}

void ImageProcesser::CImageToMat(CImage & cimage, Mat & mat)
{
	if (true == cimage.IsNull())
	{
		return;
	}


	int nChannels = cimage.GetBPP() / 8;
	if ((1 != nChannels) && (3 != nChannels))
	{
		return;
	}
	int nWidth = cimage.GetWidth();
	int nHeight = cimage.GetHeight();
	cout << nWidth << nHeight << endl;

	//�ؽ�mat  
	if (1 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC1);
	}
	else if (3 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC3);
	}


	//��������  


	uchar* pucRow;                                  //ָ������������ָ��  
	uchar* pucImage = (uchar*)cimage.GetBits();     //ָ����������ָ��  
	int nStep = cimage.GetPitch();                  //ÿ�е��ֽ���,ע���������ֵ�����и�  


	for (int nRow = 0; nRow < nHeight; nRow++)
	{
		pucRow = (mat.ptr<uchar>(nRow));
		for (int nCol = 0; nCol < nWidth; nCol++)
		{
			if (1 == nChannels)
			{
				pucRow[nCol] = *(pucImage + nRow * nStep + nCol);
			}
			else if (3 == nChannels)
			{
				for (int nCha = 0; nCha < 3; nCha++)
				{
					pucRow[nCol * 3 + nCha] = *(pucImage + nRow * nStep + nCol * 3 + nCha);
				}
			}
		}
	}
}

ImageProcesser::~ImageProcesser()
{
	delete this->mat;
}

CImage* ImageProcesser::go()
{
	if (this->cstr == "��������") {
		this->mat = this->salt(this->mat, 10000);
	}
	else if (this->cstr=="��ֵ�˲�"){
		this->mat = this->medianBlur(this->mat, 5);
	}
	this->img = new CImage();
	this->MatToCImage(*this->mat, *this->img);
	return this->img;
}

Mat * ImageProcesser::salt(Mat * mat, int n)
{
	// ��Ϊ���������ȽϿ죬�Ͳ���openmp��
	#pragma omp parallel for num_threads(threadNum)
	for (int i = 0; i < n; ++i) {
		int x1,x2, y1,y2;
		x1 = rand() % mat->rows;
		y1 = rand() % mat->cols;
		x2 = rand() % mat->rows;
		y2 = rand() % mat->cols;
		// һ����ɫͨ���ĻҶ�ͼ
		if (mat->type() == CV_8UC1) {
			mat->at<uchar>(x1, y1) = 255;
			mat->at<uchar>(x2, y2) = 0;
		}
		// ������ɫͨ���ĻҶ�ͼ
		else if (mat->type() == CV_8UC3) {
			mat->at<Vec3b>(x1, y1) = Vec3b{ 255,255,255 };
			mat->at<Vec3b>(x2, y2) = Vec3b{ 255,255,255 };
		}
	}
	return mat;
}

Mat * ImageProcesser::medianBlur(Mat * mat, int n)
{
	//��ʽ�� R * 0.144+ 0.587*G +B* 0.299 
	auto distMat = new Mat(mat->rows, mat->cols, mat->type());
	auto within = [&](int x,int y) {
		return x >= 0 && y >= 0 && x < (mat->rows) && y < (mat->cols);
	};
	int allRow = ceil(mat->rows / n);
	#pragma omp parallel for num_threads(threadNum)
	/************************************************************************/
	/* allRow�ĵط��������ñ��ʽ����Ȼ����ʱȷʵ�ǹ̶�ֵ��23333          */
	/************************************************************************/
	for (int row = 0; row < allRow; ++row) {
		for (int col = 0; col < ceil(mat->cols / n); ++col) {
			vector<pair<float,pair<int,int>>> vec;
			int initx = row*n, inity = col*n;
			for (int x = initx; x < initx+n; ++x) {
				for (int y = inity; y < inity+n; ++y) {
					if (within(x,y)) vec.emplace_back(make_pair(0.144*mat->at<Vec3b>(x, y)[0] + 0.587*mat->at<Vec3b>(x, y)[1] + 0.299*mat->at<Vec3b>(x, y)[2], make_pair(x,y)));
				}
			}
			sort(vec.begin(), vec.end());
			auto mid = vec.size() / 2;
			auto actx = vec[mid].second.first , acty = vec[mid].second.second;
			auto pointColor = mat->at<Vec3b>(actx, acty);
			for (int x = initx; x < initx + n; ++x) {
				for (int y = inity; y < inity + n; ++y) {
					if (within(x,y)) distMat->at<Vec3b>(x,y) = pointColor;
				}
			}
		}
	}
	return distMat;
}
