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

class Tensor4
{
public:
	Tensor4(const Dim4& dim)
	{
		m_dim = dim;
		m_size = (size_t)dim.dim[3];
		m_strides.dim[3] = 1;
		for (int i = 2; i >= 0; i--)
		{
			m_size *= (size_t)dim.dim[i];
			m_strides.dim[i] = m_strides.dim[i + 1] * dim.dim[i + 1];
		}
		cudaMalloc(&m_data, sizeof(float) * m_size);

		int dim_nchw[4] = { dim.dim[0], dim.dim[3], dim.dim[1], dim.dim[2] };
		int stride_nchw[4] = { m_strides.dim[0], m_strides.dim[3], m_strides.dim[1], m_strides.dim[2] };

		cudnnCreateTensorDescriptor(&m_desc);
		cudnnSetTensorNdDescriptor(m_desc, CUDNN_DATA_FLOAT, 4, dim_nchw, stride_nchw);
	}

	~Tensor4()
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
	Dim4 m_dim;
	Dim4 m_strides;
	size_t m_size;
	float* m_data;
	cudnnTensorDescriptor_t m_desc;
};

class FilterWeights4
{
public:
	FilterWeights4(const Dim4& dim)
	{
		m_dim = dim;
		m_size = 1;
		for (int i = 0; i < 4; i++)
			m_size *= dim.dim[i];

		int dim_kcrs[4] = { dim.dim[0], dim.dim[3], dim.dim[1], dim.dim[2] };

		cudaMalloc(&m_data, sizeof(float) * m_size);
		cudnnCreateFilterDescriptor(&m_desc);
		cudnnSetFilterNdDescriptor(m_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 4, dim_kcrs);
	}

	virtual ~FilterWeights4()
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
	Dim4 m_dim;
	size_t m_size;
	float* m_data;
	cudnnFilterDescriptor_t m_desc;
};
