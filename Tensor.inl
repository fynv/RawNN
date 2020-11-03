#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#ifndef _DEF_DIM
#define _DEF_DIM
template<int D>
struct Dim
{
	int dim[D];
};
typedef Dim<2> Dim2;
typedef Dim<4> Dim4;
#endif

template<int D>
class Tensor
{
public:
	Tensor(const Dim<D>& dim)
	{
		m_dim = dim;
		m_size = (size_t)dim.dim[D-1];
		m_strides.dim[D - 1] = 1;
		for (int i = D - 2; i >= 0; i--)
		{
			m_size *= (size_t)dim.dim[i];
			m_strides.dim[i] = m_strides.dim[i + 1] * dim.dim[i + 1];
		}
		cudaMalloc(&m_data, sizeof(float) * m_size);

		cudnnCreateTensorDescriptor(&m_desc);
		cudnnSetTensorNdDescriptor(m_desc, CUDNN_DATA_FLOAT, D, m_dim.dim, m_strides.dim);
	}

	~Tensor()
	{
		cudnnDestroyTensorDescriptor(m_desc);
		cudaFree(m_data);
	}

	const int* dim() const { return m_dim.dim; }
	const int* strides() const { return m_strides.dim; }
	size_t size() const { return m_size; }	
	float* data() const { return m_data; }
	cudnnTensorDescriptor_t desc() const { return m_desc; }

	void from_host(const float* hdata)
	{
		cudaMemcpy(m_data, hdata, sizeof(float) * m_size, cudaMemcpyHostToDevice);
	}

	void to_host(float* hdata) const
	{
		cudaMemcpy(hdata, m_data, sizeof(float) * m_size, cudaMemcpyDeviceToHost);
	}

private:
	Dim<D> m_dim;
	Dim<D> m_strides;
	size_t m_size;
	float* m_data;
	cudnnTensorDescriptor_t m_desc;
};

typedef Tensor<4> Tensor4;

template<int D>
class FilterWeights
{
public:
	FilterWeights(const Dim<D>& dim)
	{
		m_dim = dim;
		m_size = 1;
		for (int i = 0; i < D; i++)
			m_size *= dim.dim[i];

		cudaMalloc(&m_data, sizeof(float) * m_size);
		cudnnCreateFilterDescriptor(&m_desc);
		cudnnSetFilterNdDescriptor(m_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, D, m_dim.dim);
	}

	~FilterWeights()
	{
		cudnnDestroyFilterDescriptor(m_desc);
		cudaFree(m_data);
	}

	const int* dim() const { return m_dim.dim; }
	size_t size() const { return m_size; }
	float* data() const { return m_data; }
	cudnnFilterDescriptor_t desc() const { return m_desc; }

	void from_host(const float* hdata)
	{
		cudaMemcpy(m_data, hdata, sizeof(float) * m_size, cudaMemcpyHostToDevice);
	}

	void to_host(float* hdata) const
	{
		cudaMemcpy(hdata, m_data, sizeof(float) * m_size, cudaMemcpyDeviceToHost);
	}

private:
	Dim<D> m_dim;
	size_t m_size;
	float* m_data;
	cudnnFilterDescriptor_t m_desc;
};

typedef FilterWeights<4> FilterWeights4;