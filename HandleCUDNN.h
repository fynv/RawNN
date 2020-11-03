#pragma once

#include <cudnn.h>

class HandleCUDNN
{
public:
	static cudnnHandle_t handle();


private:
	HandleCUDNN();
	~HandleCUDNN();
	cudnnHandle_t m_handle;

};