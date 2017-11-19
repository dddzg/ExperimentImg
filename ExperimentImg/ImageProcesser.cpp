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


	//重建cimage  
	cimage.Destroy();
	cimage.Create(nWidth, nHeight, 8 * nChannels);


	//拷贝数据  


	uchar* pucRow;                                  //指向数据区的行指针  
	uchar* pucImage = (uchar*)cimage.GetBits();     //指向数据区的指针  
	int nStep = cimage.GetPitch();                  //每行的字节数,注意这个返回值有正有负  


	if (1 == nChannels)                             //对于单通道的图像需要初始化调色板  
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

	//重建mat  
	if (1 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC1);
	}
	else if (3 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC3);
	}


	//拷贝数据  


	uchar* pucRow;                                  //指向数据区的行指针  
	uchar* pucImage = (uchar*)cimage.GetBits();     //指向数据区的指针  
	int nStep = cimage.GetPitch();                  //每行的字节数,注意这个返回值有正有负  


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
	if (this->cstr == "椒盐噪声") {
		this->mat = this->salt(this->mat, 10000);
	}
	else if (this->cstr == "中值滤波"){
		this->mat = this->medianBlur(this->mat, 5);
	}
	else if (this->cstr == "双三阶插值（缩放）") {
		this->mat = this->scale(this->mat, 2);
	}
	else if (this->cstr == "双三阶插值（旋转）") {
		this->mat = this->rotate(this->mat, 45);
	}
	else if (this->cstr == "自动白平衡") {
		this->mat = this->autoBalance(this->mat);
	}
	else if (this->cstr == "自动色阶") {
		this->mat = this->autoLevel(this->mat);
	}
	else if (this->cstr == "自适应双边滤波") {
		this->mat = this->bilateralFilter(this->mat,4,25,50);
	}
	this->img = new CImage();
	this->MatToCImage(*this->mat, *this->img);
	return this->img;
}

Mat * ImageProcesser::salt(Mat * mat, int n)
{
	// 因为椒盐噪声比较快，就不加openmp了
	#pragma omp parallel for num_threads(threadNum)
	for (int i = 0; i < n; ++i) {
		int x1,x2, y1,y2;
		x1 = rand() % mat->rows;
		y1 = rand() % mat->cols;
		x2 = rand() % mat->rows;
		y2 = rand() % mat->cols;
		// 一个颜色通道的灰度图
		if (mat->type() == CV_8UC1) {
			mat->at<uchar>(x1, y1) = 255;
			mat->at<uchar>(x2, y2) = 0;
		}
		// 三个颜色通道的灰度图
		else if (mat->type() == CV_8UC3) {
			mat->at<Vec3b>(x1, y1) = Vec3b{ 255,255,255 };
			mat->at<Vec3b>(x2, y2) = Vec3b{ 255,255,255 };
		}
	}
	return mat;
}

Mat * ImageProcesser::medianBlur(Mat * mat, int n)
{
	//公式： R * 0.144+ 0.587*G +B* 0.299 
	auto distMat = new Mat(mat->rows, mat->cols, mat->type());
	int allRow = ceil(mat->rows / n);
	#pragma omp parallel for num_threads(threadNum)
	/************************************************************************/
	/* allRow的地方好像不能用表达式，虽然运行时确实是固定值。23333          */
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

// 因为展示的原因，原图需要比较小比较好
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
			float w_x[4], w_y[4];//行列方向的加权系数
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

// 逆时针旋转，注意坐标系的变换
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
			// 换坐标原点，旋转，再换回来，也可以理解为平移旋转矩阵相乘。
			float x = (i - newRow / 2)*cos(radian) + (j - newCol / 2)*sin(radian) + mat->rows / 2;
			float y = -(i - newRow / 2)*sin(radian) + (j - newCol / 2)*cos(radian) + mat->cols / 2;
			float w_x[4], w_y[4];//行列方向的加权系数
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
	// 不想理偶数核
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