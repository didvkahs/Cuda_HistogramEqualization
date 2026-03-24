#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "typedef.h"
#include "BmpReader.h"

#include "Histogram.h"
#include <iostream>

constexpr int BLOCK = 16;
constexpr int MAX_PIXEL_VALUE = 256;
constexpr int HISTOGRAM_SIZE = MAX_PIXEL_VALUE * sizeof(uint32_t);

inline bool CudaCheck(cudaError_t err);
inline bool CudaKernelCheck(void);

__global__ void histogram(uint32_t* histogramVal, uint8_t* buf, size_t pitch, uint32_t width, uint32_t height);
__global__ void equalization(uint32_t* hitogramVal, uint8_t* buf, size_t pitch, uint32_t width, uint32_t height);

Histogram::Histogram(void) {};
Histogram::~Histogram(void) {};


bool Histogram::Initialize(void)
{
	if (!Bmp::LoadRGBs(&h_rBuf, &h_gBuf, &h_bBuf, width, height))
	{
		fprintf(stderr, "LoadRGBs failed with error \n");
		goto LB_FAILED_LOAD_RGB;
	}

	h_Histo= new uint32_t[MAX_PIXEL_VALUE];
	memset(h_Histo, 0, HISTOGRAM_SIZE);

	if (!CudaCheck(cudaMalloc(&d_Histo, HISTOGRAM_SIZE)))
	{
		delete[] h_Histo;
		goto LB_FAILED_ALLOC_HISTOGRAM;
	}
	if (!CudaCheck(cudaMemcpy(d_Histo, h_Histo, HISTOGRAM_SIZE, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_ALLOC_HISTOGRAM;
	}

	if (!CudaCheck(cudaMallocPitch(&d_r, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
	{
		goto LB_FAILED_ALLOC_DEVMEM1;
	}
	if (!CudaCheck(cudaMallocPitch(&d_g, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
	{
		cudaFree(d_r);
		goto LB_FAILED_ALLOC_DEVMEM1;
	}
	if (!CudaCheck(cudaMallocPitch(&d_b, &pitch, static_cast<size_t>(width), static_cast<size_t>(height))))
	{
		cudaFree(d_r);
		cudaFree(d_g);
		goto LB_FAILED_ALLOC_DEVMEM1;
	}

	if (!CudaCheck(cudaMemcpy2D(d_r, pitch, h_rBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_ALLOC_DEVMEN2;
	}
	if (!CudaCheck(cudaMemcpy2D(d_g, pitch, h_gBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_ALLOC_DEVMEN2;
	}
	if (!CudaCheck(cudaMemcpy2D(d_b, pitch, h_bBuf->data, sizeof(uint8_t) * width, sizeof(uint8_t) * width, height, cudaMemcpyHostToDevice)))
	{
		goto LB_FAILED_ALLOC_DEVMEN2;
	}


	return true;

	LB_FAILED_ALLOC_DEVMEN2:
	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);

	LB_FAILED_ALLOC_DEVMEM1:
	delete[] h_Histo;
	cudaFree(d_Histo);

	LB_FAILED_ALLOC_HISTOGRAM:
	Bmp::FreeMem(&h_rBuf);
	Bmp::FreeMem(&h_gBuf);
	Bmp::FreeMem(&h_bBuf);

	LB_FAILED_LOAD_RGB:
	return false;
}

void Histogram::ProcessImage(void)
{	
	if (!getHistogram())
	{
		fprintf(stderr, "histogram caculation failed\n");
		return;
	}
	if (!applyEqualization())
	{
		fprintf(stderr, "applying equalization failed \n");
		return;
	}
}

void Histogram::CloseBufHandles(void)
{
	if (h_rBuf) Bmp::FreeMem(&h_rBuf);
	if (h_gBuf) Bmp::FreeMem(&h_gBuf);
	if (h_bBuf) Bmp::FreeMem(&h_bBuf);

	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
}






bool Histogram::getHistogram(void)
{
	dim3 block(BLOCK, BLOCK);
	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
	histogram << <grid, block >> > (d_Histo, d_r, pitch, width, height);
	if (!CudaKernelCheck()) { return false; }

	cudaMemcpy(h_Histo, d_Histo, HISTOGRAM_SIZE, cudaMemcpyDeviceToHost);

	return true;
}

bool Histogram::applyEqualization(void)
{
	const uint32_t DENOMINATOR = width * height - 1;

	float cdf = 0.0f; // to cumulate pixel value

	for (int i = 0; i < MAX_PIXEL_VALUE; ++i)
	{
		cdf += h_Histo[i]; // cumulate count 
		float h = (cdf / DENOMINATOR) * 255.0f;

		h_Histo[i] = (uint8_t)(h + 0.5f); // if you multiple to max value you can get normal distribution value
	}

	cudaMemcpy(d_Histo, h_Histo, HISTOGRAM_SIZE, cudaMemcpyHostToDevice);

	dim3 block(BLOCK, BLOCK);
	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);

	equalization << <grid, block >> > (d_Histo, d_r, pitch, width, height);
	if (!CudaKernelCheck()) { return false; }
	equalization << <grid, block >> > (d_Histo, d_g, pitch, width, height);
	if (!CudaKernelCheck()) { return false; }
	equalization << <grid, block >> > (d_Histo, d_b, pitch, width, height);
	if (!CudaKernelCheck()) { return false; }

	cudaMemcpy2D(h_rBuf->data, width, d_r, pitch, width, height, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(h_gBuf->data, width, d_g, pitch, width, height, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(h_bBuf->data, width, d_b, pitch, width, height, cudaMemcpyDeviceToHost);

	Bmp::StoreRGBs("lennaEqualized.bmp", width, height, &h_rBuf, &h_gBuf, &h_bBuf);

	return true;
}




__global__ void histogram(uint32_t* histogramVal, uint8_t* buf, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width || srcY >= height || srcX < 0 || srcY < 0) return;

	uint8_t pixelVal = buf[srcX + pitch * srcY];
	
	atomicAdd(&histogramVal[pixelVal], 1);
}

__global__ void equalization(uint32_t* histogramVal, uint8_t* buf, size_t pitch, uint32_t width, uint32_t height)
{
	const uint32_t srcX = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

	if (srcX >= width || srcY >= height || srcX < 0 || srcY < 0) return;

	uint8_t pixelVal = buf[srcX + pitch * srcY];
	uint8_t equalizedVal = histogramVal[pixelVal];

	buf[srcX + pitch * srcY] = equalizedVal;
}





inline bool CudaCheck(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Error : " << cudaGetErrorString(err) << "at line " << __LINE__ << "\n";
		return false;
	}
	return true;
}

inline bool CudaKernelCheck(void)
{
	return CudaCheck(cudaGetLastError());
}