#include "util.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
// 双三阶线性插值的定义。由于cuda里只能调用__global__的函数，所以只能复制过来了。。
__constant__ float a = -0.5;
__device__ void getW_x(float w_x[4], float x) {
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
}
__device__ void getW_y(float w_y[4], float y) {
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
}
__global__ void scale(uchar3* src, uchar3* dist,int srcRow,int srcCol,int distRow,int distCol,float n) {
	auto distX = blockDim.x*blockIdx.x + threadIdx.x;
	auto distY = blockDim.y*blockIdx.y + threadIdx.y;
	if (!(distX < distRow && distY < distCol)) return;
	auto distPos = (distY + distX * distCol);
	auto srcX = distX /2;
	auto srcY = distY /2;
	auto withinSrc = [&](int x,int y) {
		return x >= 0 && y >= 0 && x < srcRow && y < srcCol;
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
				auto pos = yy + xx*srcCol;
				temp.x += src[pos].x * w_x[i] * w_y[j];
				temp.y += src[pos].y * w_x[i] * w_y[j];
				temp.z += src[pos].z * w_x[i] * w_y[j];
			}
		}
	}

	dist[distPos] = uchar3{temp.x,temp.y,temp.z};
}
__global__ void rotate(uchar3* src, uchar3* dist, int srcRow, int srcCol, int distRow, int distCol, float radian) {
	auto distX = blockDim.x*blockIdx.x + threadIdx.x;
	auto distY = blockDim.y*blockIdx.y + threadIdx.y;
	if (!(distX < distRow && distY < distCol && distX>=0 && distY>=0)) return;
	auto distPos = (distY + distX * distCol);
	// 谜之/2 *0.5
	float srcX = (distX - distRow*0.5)*cosf(radian) + (distY - distCol*0.5)*sinf(radian) + srcRow*0.5;
	float srcY = -(distX - distRow*0.5)*sinf(radian) + (distY - distCol*0.5)*cosf(radian) + srcCol*0.5;
	auto withinSrc = [&](int x, int y) {
		return x >= 0 && y >= 0 && x < srcRow && y < srcCol;
	};

	float w_x[4], w_y[4];//行列方向的加权系数
	getW_x(w_x, srcX);
	getW_y(w_y, srcY);
	float3 temp = { 0,0,0 };
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			int xx = int(srcX) + i - 1;
			int yy = int(srcY) + j - 1;
			if (withinSrc(xx, yy)) {
				auto pos = yy + xx*srcCol;
				temp.x += src[pos].x * w_x[i] * w_y[j];
				temp.y += src[pos].y * w_x[i] * w_y[j];
				temp.z += src[pos].z * w_x[i] * w_y[j];
			}
		}
	}
	dist[distPos] = uchar3{ min(temp.x,255.0f),min(temp.y,255.0f),min(temp.z,255.0f) };
}
__global__ void merge(uchar3* src, uchar3* dist, int srcRow, int srcCol, int distRow, int distCol, float alpha) {
	auto distX = blockDim.x*blockIdx.x + threadIdx.x;
	auto distY = blockDim.y*blockIdx.y + threadIdx.y;
	auto distPos = distY + distX * distCol;
	auto srcPos = distY + distX * srcCol;
	if (distX < srcRow && distY < srcCol) {
		dist[distPos].x = min(int(alpha*src[srcPos].x + (1 - alpha)*dist[distPos].x), 255);
		dist[distPos].y = min(int(alpha*src[srcPos].y + (1 - alpha)*dist[distPos].y), 255);
		dist[distPos].z = min(int(alpha*src[srcPos].z + (1 - alpha)*dist[distPos].z), 255);
	}
}
extern "C" Mat* scaleUseCuda(Mat* mat, float n) {
	auto rows = int(mat->rows*n), cols = int(mat->cols*n);
	auto distMat = new Mat(rows,cols,mat->type());
	auto srcSize = (mat->rows)*(mat->cols) * sizeof(uchar3);
	int distSize = rows*cols*sizeof(uchar3);
	uchar3* gpuSrcMat = nullptr;
	uchar3* gpuDistMat = nullptr;
	cudaMalloc((void **)&gpuSrcMat, srcSize);
	cudaMalloc((void **)&gpuDistMat, distSize);
	cudaMemcpy(gpuSrcMat, mat->data, srcSize, cudaMemcpyHostToDevice);

	dim3 dimBlock(32, 32); // max 1024
	dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y -1) / dimBlock.y); 
	scale << <dimGrid, dimBlock >> > (gpuSrcMat, gpuDistMat, mat->rows, mat->cols, rows, cols, n);
	cudaMemcpy(distMat->data, gpuDistMat, distSize, cudaMemcpyDeviceToHost);
	cudaFree(gpuSrcMat);
	cudaFree(gpuDistMat);
	return distMat;
}
extern "C" Mat* rotateUseCuda(Mat* mat, float angle) {
	auto radian = util::toRadian(angle);
	int srcRows = mat->rows, srcCols = mat->cols;
	//int newRow = mat->rows*cos(radian) + mat->cols*sin(radian), newCol = mat->rows*sin(radian) + mat->cols*cos(radian);
	int distRows = srcRows*cos(radian) + srcCols*sin(radian), distCols = srcRows*sin(radian) + srcCols*cos(radian);
	auto distMat = new Mat(distRows, distCols, mat->type());
	auto srcSize = srcRows*srcCols * sizeof(uchar3);
	auto distSize = distRows*distCols * sizeof(uchar3);
	uchar3* gpuSrcMat = nullptr;
	uchar3* gpuDistMat = nullptr;
	cudaMalloc((void **)&gpuSrcMat, srcSize);
	cudaMalloc((void **)&gpuDistMat, distSize);
	cudaMemcpy(gpuSrcMat, mat->data, srcSize, cudaMemcpyHostToDevice);
	dim3 dimBlock(32, 32); // max 1024
	dim3 dimGrid((distRows + dimBlock.x - 1) / dimBlock.x, (distCols + dimBlock.y - 1) / dimBlock.y);
	rotate << <dimGrid, dimBlock >> > (gpuSrcMat, gpuDistMat, srcRows, srcCols, distRows, distCols, radian);
	cudaMemcpy(distMat->data, gpuDistMat, distSize, cudaMemcpyDeviceToHost);
	cudaFree(gpuSrcMat);
	cudaFree(gpuDistMat);
	return distMat;
}
extern "C" Mat* mergeUseCuda(Mat* srcMat, Mat* distMat, float alpha) {
	uchar3* gpuSrcMat = nullptr;
	uchar3* gpuDistMat = nullptr;
	auto srcSize = srcMat->rows*srcMat->cols * sizeof(uchar3);
	auto distSize = distMat->rows*distMat->cols * sizeof(uchar3);
	cudaMalloc((void **)&gpuSrcMat, srcSize);
	cudaMalloc((void **)&gpuDistMat, distSize);
	cudaMemcpy(gpuSrcMat, srcMat->data, srcSize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuDistMat, distMat->data, distSize, cudaMemcpyHostToDevice);
	dim3 dimBlock(32, 32); // max 1024
	dim3 dimGrid((distMat->rows + dimBlock.x - 1) / dimBlock.x, (distMat->cols + dimBlock.y - 1) / dimBlock.y);
	merge << <dimGrid, dimBlock >> > (gpuSrcMat, gpuDistMat, srcMat->rows, srcMat->cols, distMat->rows, distMat->cols, alpha);
	cudaMemcpy(distMat->data, gpuDistMat, distSize, cudaMemcpyDeviceToHost);
	cudaFree(gpuSrcMat);
	cudaFree(gpuDistMat);
	return distMat;
}