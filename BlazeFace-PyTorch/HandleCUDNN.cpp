#include "HandleCUDNN.h"

HandleCUDNN::HandleCUDNN()
{
	cudnnCreate(&m_handle);
}

HandleCUDNN::~HandleCUDNN()
{
	cudnnDestroy(m_handle);
}

cudnnHandle_t HandleCUDNN::handle()
{
	static HandleCUDNN singlton;
	return singlton.m_handle;
}
