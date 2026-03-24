#pragma once
#include <cstdint>
#include "typedef.h"


class Histogram
{
public:

	Histogram(void);
	~Histogram(void);

	bool Initialize(void);
	void ProcessImage(void);
	void CloseBufHandles(void);

private:
	bool getHistogram(void);
	bool applyEqualization(void);

private:

	uint32_t* h_Histo = nullptr;
	uint32_t* d_Histo = nullptr;

	uint32_t width = 0;
	uint32_t height = 0;
	Buf_s* h_rBuf = nullptr;
	Buf_s* h_gBuf = nullptr;
	Buf_s* h_bBuf = nullptr;

	size_t pitch = 0;
	uint8_t* d_r = nullptr;
	uint8_t* d_g = nullptr;
	uint8_t* d_b = nullptr;
};