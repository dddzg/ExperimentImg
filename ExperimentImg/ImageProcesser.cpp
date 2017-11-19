#include "stdafx.h"
#include "ImageProcesser.h"


ImageProcesser::ImageProcesser(CImage * img, const CString & cstr, int threadNum, bool useGPU)
{
	this->initImg = img;
	this->mat = new Mat();
	this->cstr = move(cstr);
	this->threadNum = threadNum;
	this->useGPU = useGPU;
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
	else if (this->cstr == "��ֵ�˲�"){
		this->mat = this->medianBlur(this->mat, 5);
	}
	else if (this->cstr == "˫���ײ�ֵ�����ţ�") {
		this->mat = this->scale(this->mat, 2);
	}
	else if (this->cstr == "˫���ײ�ֵ����ת��") {
		this->mat = this->rotate(this->mat, 45);
	}
	else if (this->cstr == "�Զ���ƽ��") {
		this->mat = this->autoBalance(this->mat);
	}
	else if (this->cstr == "�Զ�ɫ��") {
		this->mat = this->autoLevel(this->mat);
	}
	else if (this->cstr == "����Ӧ˫���˲�") {
		this->mat = this->bilateralFilter(this->mat,4,25,50);
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
					if (within(mat,x,y)) vec.emplace_back(make_pair(0.144*mat->at<Vec3b>(x, y)[0] + 0.587*mat->at<Vec3b>(x, y)[1] + 0.299*mat->at<Vec3b>(x, y)[2], make_pair(x,y)));
				}
			}
			sort(vec.begin(), vec.end());
			auto mid = vec.size() / 2;
			auto actx = vec[mid].second.first , acty = vec[mid].second.second;
			auto pointColor = mat->at<Vec3b>(actx, acty);
			for (int x = initx; x < initx + n; ++x) {
				for (int y = inity; y < inity + n; ++y) {
					if (within(mat,x,y)) distMat->at<Vec3b>(x,y) = pointColor;
				}
			}
		}
	}
	return distMat;
}

// ��Ϊչʾ��ԭ��ԭͼ��Ҫ�Ƚ�С�ȽϺ�
Mat * ImageProcesser::scale(Mat * mat, float n)
{
	if (this->useGPU) { 
		return scaleUseCuda(mat,n); 
	}
	int newRow = mat->rows*n, newCol = mat->cols*n;
	auto bigMat = new Mat(newRow, newCol, mat->type());
	#pragma omp parallel for num_threads(threadNum)
	for (int i = 0; i < newRow; i++) {
		for (int j = 0; j < newCol; j++) {
			float x = i/n;
			float y = j/n;
			float w_x[4], w_y[4];//���з���ļ�Ȩϵ��
			getW_x(w_x, x);
			getW_y(w_y, y);
			Vec3f temp = { 0, 0, 0 };
			for (int s = 0; s < 4; s++) {
				for (int t = 0; t < 4; t++) {
					if (within(mat,int(x) + s - 1, int(y) + t - 1))
					temp += (Vec3f)(mat->at<Vec3b>(int(x) + s - 1, int(y) + t - 1))*w_x[s] * w_y[t];
				}
			}
			bigMat->at<Vec3b>(i, j) = move((Vec3b)temp);
		}
	}
	return bigMat;
}

// ��ʱ����ת��ע������ϵ�ı任
Mat * ImageProcesser::rotate(Mat * mat, float angle)
{
	if (this->useGPU) {
		return rotateUseCuda(mat, angle);
	}
	auto radian = toRadian(angle);
	int newRow = mat->rows*cos(radian) + mat->cols*sin(radian), newCol = mat->rows*sin(radian) + mat->cols*cos(radian);
	auto bigMat = new Mat(newRow, newCol, mat->type());
	#pragma omp parallel for num_threads(threadNum)
	for (int i = 0; i < newRow; i++) {
		for (int j = 0; j < newCol; j++) {
			// ������ԭ�㣬��ת���ٻ�������Ҳ�������Ϊƽ����ת������ˡ�
			float x = (i - newRow / 2)*cos(radian) + (j - newCol / 2)*sin(radian) + mat->rows / 2;
			float y = -(i - newRow / 2)*sin(radian) + (j - newCol / 2)*cos(radian) + mat->cols / 2;
			float w_x[4], w_y[4];//���з���ļ�Ȩϵ��
			getW_x(w_x, x);
			getW_y(w_y, y);
			Vec3f temp = { 0, 0, 0 };
			for (int s = 0; s < 4; s++) {
				for (int t = 0; t < 4; t++) {
					if (within(mat,int(x) + s - 1, int(y) + t - 1))
						temp += (Vec3f)(mat->at<Vec3b>(int(x) + s - 1, int(y) + t - 1))*w_x[s] * w_y[t];
				}
			}
			bigMat->at<Vec3b>(i, j) = move((Vec3b)temp);
		}
	}
	return bigMat;
}

Mat * ImageProcesser::autoBalance(Mat * mat)
{
	double R = 0, G = 0, B = 0;
	for (int x = 0; x < mat->rows; ++x) {
		for (int y = 0; y < mat->cols; ++y) {
			auto point = mat->at<Vec3b>(x, y);
			R += point[0];
			G += point[1];
			B += point[2];
		}
	}
	double KR = (R + G + B) / (3 * R), KG = (R + G + B) / (3 * G), KB = (R + G + B) / (3 * B);
	for (int x = 0; x < mat->rows; ++x) {
		for (int y = 0; y < mat->cols; ++y) {
			auto& point = mat->at<Vec3b>(x, y);
			point[0] = min(255, int(point[0] * KR));
			point[1] = min(255, int(point[1] * KG));
			point[2] = min(255, int(point[2] * KB));
		}
	}
	return this->mat;
}

Mat * ImageProcesser::autoLevel(Mat * mat)
{
	int maxx[3] = { 0,0,0 }, minn[3] = { 255,255,255 };
	for (int x = 0; x < mat->rows; ++x) {
		for (int y = 0; y < mat->cols; ++y) {
			auto& point = mat->at<Vec3b>(x, y);
			for (auto i = 0; i < 3; ++i) {
				if (point[i] > maxx[i]) {
					maxx[i] = point[i];
				}
				if (point[i] < minn[i]) {
					minn[i] = point[i];
				}
			}
		}
	}
	for (int x = 0; x < mat->rows; ++x) {
		for (int y = 0; y < mat->cols; ++y) {
			auto& point = mat->at<Vec3b>(x, y);
			for (auto i = 0; i < 3; ++i) {
				if (maxx[i]!=minn[i]){
					point[i] = float(point[i] - minn[i]) / (maxx[i] - minn[i]) * 255;
				}
			}
		}
	}
	return mat;
}

Mat * ImageProcesser::bilateralFilter(Mat * mat, int d, double sigmaColor, double sigmaSpace)
{
	auto spaceFunction = [&](int x, int y, int xx, int yy) {
		return -((x - xx)*(x - xx) + (y - yy)*(y - yy)) / (2 * sigmaSpace*sigmaSpace);
	};
	auto colorFunction = [&](int a,int b) {
		return -(a-b)*(a-b) / (2 * sigmaColor*sigmaColor);
	};
	auto distMat = new Mat(mat->rows, mat->cols, mat->type());
	// ������ż����
	if (d % 2 == 0) d += 1;
	#pragma omp parallel for num_threads(threadNum)
	for (int x = 0; x < mat->rows; ++x) {
		for (int y = 0; y < mat->cols; ++y) {
			auto& centerPoint = mat->at<Vec3b>(x, y);
			double sums[3] = { 0,0,0 };
			double color[3] = { 0,0,0 };
			for (int xx = x - d / 2; xx <= x + d / 2; ++xx) {
				for (int yy = y - d / 2; yy <= y + d / 2; ++yy) {
					if (within(mat,xx, yy)) {
						auto& point = mat->at<Vec3b>(xx, yy);
						for (int i = 0; i < 3; ++i) {
							double d = spaceFunction(x, y, xx, yy);
							double r = colorFunction(point[i], centerPoint[i]);
							double w = exp(d + r);
							color[i] += point[i] * w;
							sums[i] += w;
						}
					}
				}
			}
			for (int i = 0; i < 3; ++i) {
				color[i] /= sums[i];
				color[i] = max(0, int(color[i]));
				color[i] = min(255, int(color[i]));
			}
			distMat->at<Vec3b>(x, y) = Vec3b{uchar(color[0]),uchar(color[1]),uchar(color[2])};
		}
	}
	return distMat;
}

CImage * ImageProcesser::merge(CImage * src, CImage * dist,double alpha, bool useGPU)
{
	auto srcMat = new Mat(), distMat = new Mat();
	CImageToMat(*src, *srcMat);
	CImageToMat(*dist, *distMat);
	if (useGPU) {
		distMat=mergeUseCuda(srcMat,distMat,alpha);
	} else{
		auto within = [&](int x, int y) {
			return x >= 0 && y >= 0 && x < (srcMat->rows) && y < (srcMat->cols);
		};
		for (int x = 0; x < distMat->rows; ++x) {
			for (int y = 0; y < distMat->cols; ++y) {
				if (within(x, y)) {
					auto &srcPoint = srcMat->at<Vec3b>(x, y);
					auto &distPoint = distMat->at<Vec3b>(x, y);
					for (int i = 0; i < 3; ++i) {
						distPoint[i] = min(int(alpha*srcPoint[i] + (1 - alpha)*distPoint[i]), 255);
					}
				}
			}
		}
	}
	auto img = new CImage();
	MatToCImage(*distMat, *img);
	return img;
}