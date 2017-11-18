#include "util.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
using namespace util;
__global__ void scale(uchar3* src, uchar3* dist,int srcRow,int srcCol,int distRow,int distCol,float n) {
	auto distX = blockDim.x*blockIdx.x + threadIdx.x;
	auto distY = blockDim.y*blockIdx.y + threadIdx.y;
	if (!(distX < distRow && distY < distCol)) return;
	auto distPos = distX + distY * distRow;
	auto srcX = int(distX /n);
	auto srcY = int(distY /n);
	auto srcPos = srcX + srcY*srcRow;
	float a = -0.5;
	auto withinSrc = [&](int x,int y) {
		return x >= 0 && y >= 0 && x < srcRow && y < srcCol;
	};
	auto withinDist = [&](int x, int y) {
		return x >= 0 && y >= 0 && x < distRow && y < distCol;
	};
	// 双三阶线性插值的定义。由于cuda里只能调用__global__的函数，所以只能复制过来了。。
	auto getW_x = [&](float w_x[4], float x) {
		int X = (int)x;
		float stemp_x[4];
		stemp_x[0] = 1 + (x - X);
		stemp_x[1] = x - X;
		stemp_x[2] = 1 - (x - X);
		stemp_x[3] = 2 - (x - X);

		w_x[0] = a*abs(stemp_x[0] * stemp_x[0] * stemp_x[0]) - 5 * a*stemp_x[0] * stemp_x[0] + 8 * a*abs(stemp_x[0]) - 4 * a;
		w_x[1] = (a + 2)*abs(stemp_x[1] * stemp_x[1] * stemp_x[1]) - (a + 3)*stemp_x[1] * stemp_x[1] + 1;
		w_x[2] = (a + 2)*abs(stemp_x[2] * stemp_x[2] * stemp_x[2]) - (a + 3)*stemp_x[2] * stemp_x[2] + 1;
		w_x[3] = a*abs(stemp_x[3] * stemp_x[3] * stemp_x[3]) - 5 * a*stemp_x[3] * stemp_x[3] + 8 * a*abs(stemp_x[3]) - 4 * a;
	};
	auto getW_y = [&](float w_y[4], float y) {
		int Y = (int)y;
		float stemp_y[4];
		stemp_y[0] = 1.0 + (y - Y);
		stemp_y[1] = y - Y;
		stemp_y[2] = 1 - (y - Y);
		stemp_y[3] = 2 - (y - Y);

		w_y[0] = a*abs(stemp_y[0] * stemp_y[0] * stemp_y[0]) - 5 * a*stemp_y[0] * stemp_y[0] + 8 * a*abs(stemp_y[0]) - 4 * a;
		w_y[1] = (a + 2)*abs(stemp_y[1] * stemp_y[1] * stemp_y[1]) - (a + 3)*stemp_y[1] * stemp_y[1] + 1;
		w_y[2] = (a + 2)*abs(stemp_y[2] * stemp_y[2] * stemp_y[2]) - (a + 3)*stemp_y[2] * stemp_y[2] + 1;
		w_y[3] = a*abs(stemp_y[3] * stemp_y[3] * stemp_y[3]) - 5 * a*stemp_y[3] * stemp_y[3] + 8 * a*abs(stemp_y[3]) - 4 * a;
	};
	float w_x[4], w_y[4];//行列方向的加权系数
	getW_x(w_x, srcX);
	getW_y(w_y, srcY);
	float3 temp = { 0,0,0 };
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			int xx = int(srcX) + i - 1;
			int yy = int(srcY) + j - 1;
			if (withinSrc(xx,yy)) {
				temp.x += src[xx + yy*srcRow].x * w_x[i] * w_y[j];
				temp.y += src[xx + yy*srcRow].y * w_x[i] * w_y[j];
				temp.z += src[xx + yy*srcRow].z * w_x[i] * w_y[j];
			}
		}
	}

	dist[distPos] = uchar3{temp.x,temp.y,temp.z};

}


extern "C" Mat* scaleUseCuda(Mat* mat, float n) {
	auto rows = int(mat->rows*n), cols = int(mat->cols*n);
	auto distMat = new Mat(rows,cols,mat->type());
	auto srcSize = (mat->rows)*(mat->cols) * sizeof(uchar3);
	int distSize = srcSize*n*n;
	uchar3* gpuSrcMat = nullptr;
	uchar3* gpuDistMat = nullptr;
	cudaMalloc((void **)&gpuSrcMat, srcSize);
	cudaMalloc((void **)&gpuDistMat, distSize);
	cudaMemcpy(gpuSrcMat, mat->data, srcSize, cudaMemcpyHostToDevice);

	dim3 dimBlock(32, 32); // max 1024
	dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y -1) / dimBlock.y); 
	scale << <dimGrid, dimBlock >> > (gpuSrcMat, gpuDistMat, mat->rows, mat->cols, rows, cols,n);
	cudaMemcpy(distMat->data, gpuDistMat, distSize, cudaMemcpyDeviceToHost);
	return distMat;
}