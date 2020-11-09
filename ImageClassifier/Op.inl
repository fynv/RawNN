#pragma once

#include "Tensor.inl"
#include "HandleCUDNN.h"

class Padding
{
public:
	Padding(const Tensor4* input, Tensor4* output, const Dim4& pad_before, const Dim4& pad_after)
		: m_input(input), m_output(output)
	{

		int before_nchw[4] = { pad_before.dim[0], pad_before.dim[3], pad_before.dim[1], pad_before.dim[2] };
		int after_nchw[4] = { pad_after.dim[0], pad_after.dim[3], pad_after.dim[1], pad_after.dim[2] };

		cudnnCreateTensorTransformDescriptor(&m_desc);
		cudnnSetTensorTransformDescriptor(m_desc, 4, CUDNN_TENSOR_NHWC, before_nchw, after_nchw, nullptr, CUDNN_TRANSFORM_FOLD);
	}

	~Padding()
	{
		cudnnDestroyTensorTransformDescriptor(m_desc);
	}

	void Run() const
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		cudnnTransformTensorEx(HandleCUDNN::handle(), m_desc,
			&alpha, m_input->desc(), m_input->data(),
			&beta, m_output->desc(), m_output->data());
	}

private:
	cudnnTensorTransformDescriptor_t m_desc;
	const Tensor4* m_input;
	Tensor4* m_output;

};

class Convolution2D
{
public:
	Convolution2D(const Tensor4* input, const FilterWeights4* weights, Tensor4* output, const Dim2& pad = { 0,0 }, const Dim2& stride = { 1,1 }, const Dim2& dilation = { 1,1 }, int group_count = 1)
		: m_input(input), m_weights(weights), m_output(output)
		, m_algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
	{
		cudnnCreateConvolutionDescriptor(&m_desc);
		cudnnSetConvolutionNdDescriptor(m_desc, 2, pad.dim, stride.dim, dilation.dim, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		if (group_count > 1)
			cudnnSetConvolutionGroupCount(m_desc, group_count);
		cudnnGetConvolutionForwardWorkspaceSize(HandleCUDNN::handle(), m_input->desc(), m_weights->desc(), m_desc, m_output->desc(), m_algo, &m_workSpaceSize);
		if (m_workSpaceSize > 0)
			cudaMalloc(&m_workSpace, m_workSpaceSize);
	}

	~Convolution2D()
	{
		if (m_workSpace != nullptr)
			cudaFree(m_workSpace);
		cudnnDestroyConvolutionDescriptor(m_desc);
	}

	void Run() const
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		cudnnConvolutionForward(HandleCUDNN::handle(),
			&alpha, m_input->desc(), m_input->data(),
			m_weights->desc(), m_weights->data(),
			m_desc, m_algo, m_workSpace, m_workSpaceSize,
			&beta, m_output->desc(), m_output->data());
	}

private:
	cudnnConvolutionDescriptor_t m_desc;
	const Tensor4* m_input;
	const FilterWeights4* m_weights;
	Tensor4* m_output;

	cudnnConvolutionFwdAlgo_t m_algo;
	size_t m_workSpaceSize = 0;
	void* m_workSpace = nullptr;

};

class Convolution2DBias
{
public:
	Convolution2DBias(const Tensor4* input, const FilterWeights4* weights, const Tensor4* bias, Tensor4* output, const Dim2& pad = { 0,0 }, const Dim2& stride = { 1,1 }, const Dim2& dilation = { 1,1 }, int group_count = 1)
		: m_input(input), m_weights(weights), m_bias(bias), m_output(output)
		, m_algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
	{
		cudnnCreateConvolutionDescriptor(&m_desc);
		cudnnSetConvolutionNdDescriptor(m_desc, 2, pad.dim, stride.dim, dilation.dim, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		if (group_count > 1)
			cudnnSetConvolutionGroupCount(m_desc, group_count);
		cudnnGetConvolutionForwardWorkspaceSize(HandleCUDNN::handle(), m_input->desc(), m_weights->desc(), m_desc, m_output->desc(), m_algo, &m_workSpaceSize);
		if (m_workSpaceSize > 0)
			cudaMalloc(&m_workSpace, m_workSpaceSize);
		cudnnCreateActivationDescriptor(&m_act_desc);
		cudnnSetActivationDescriptor(m_act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.0);
	}

	~Convolution2DBias()
	{
		cudnnDestroyActivationDescriptor(m_act_desc);
		if (m_workSpace != nullptr)
			cudaFree(m_workSpace);
		cudnnDestroyConvolutionDescriptor(m_desc);
	}

	void Run() const
	{
		float alpha = 1.0f;
		float alpha2 = 0.0f;
		cudnnConvolutionBiasActivationForward(HandleCUDNN::handle(),
			&alpha, m_input->desc(), m_input->data(),
			m_weights->desc(), m_weights->data(),
			m_desc, m_algo, m_workSpace, m_workSpaceSize,
			&alpha2, m_output->desc(), m_output->data(),
			m_bias->desc(), m_bias->data(), m_act_desc,
			m_output->desc(), m_output->data());
	}

private:
	cudnnConvolutionDescriptor_t m_desc;
	const Tensor4* m_input;
	const FilterWeights4* m_weights;
	const Tensor4* m_bias;
	Tensor4* m_output;

	cudnnConvolutionFwdAlgo_t m_algo;
	size_t m_workSpaceSize = 0;
	void* m_workSpace = nullptr;
	cudnnActivationDescriptor_t m_act_desc;

};

class BatchNormalization
{
public:
	BatchNormalization(const Tensor4* input, const Tensor4* scale, const Tensor4* bias, const Tensor4* mean, const Tensor4* variance, Tensor4* output)
		: m_input(input), m_scale(scale), m_bias(bias), m_mean(mean), m_variance(variance), m_output(output) {}
	~BatchNormalization() {}

	void Run() const
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		cudnnBatchNormalizationForwardInference(HandleCUDNN::handle(),
			CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
			m_input->desc(), m_input->data(),
			m_output->desc(), m_output->data(),
			m_scale->desc(), m_scale->data(), m_bias->data(),
			m_mean->data(), m_variance->data(), 0.001);
	}

private:
	const Tensor4* m_input;
	const Tensor4* m_scale;
	const Tensor4* m_bias;
	const Tensor4* m_mean;
	const Tensor4* m_variance;
	Tensor4* m_output;

};


class RELU
{
public:
	RELU(Tensor4* output)
		: m_output(output)
	{
		cudnnCreateActivationDescriptor(&m_desc);
		cudnnSetActivationDescriptor(m_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
	}

	~RELU()
	{
		cudnnDestroyActivationDescriptor(m_desc);
	}

	void Run() const
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		cudnnActivationForward(HandleCUDNN::handle(), m_desc,
			&alpha, m_output->desc(), m_output->data(),
			&beta, m_output->desc(), m_output->data());
	}

private:
	Tensor4* m_output;
	cudnnActivationDescriptor_t m_desc;
};


class Pooling2D
{
public:
	Pooling2D(const Tensor4* input, Tensor4* output, const Dim2& win, const Dim2& pad, const Dim2& stride)
		: m_input(input), m_output(output)
	{
		cudnnCreatePoolingDescriptor(&m_desc);
		cudnnSetPoolingNdDescriptor(m_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, win.dim, pad.dim, stride.dim);
	}

	~Pooling2D()
	{
		cudnnDestroyPoolingDescriptor(m_desc);
	}

	void Run() const
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		cudnnPoolingForward(HandleCUDNN::handle(), m_desc,
			&alpha, m_input->desc(), m_input->data(),
			&beta, m_output->desc(), m_output->data());
	}

private:
	cudnnPoolingDescriptor_t m_desc;
	const Tensor4* m_input;
	Tensor4* m_output;
};


class AddTensor
{
public:
	AddTensor(const Tensor4* input, Tensor4* output)
		: m_input(input), m_output(output) {}

	void Run() const
	{
		float alpha = 1.0f;
		cudnnAddTensor(HandleCUDNN::handle(),
			&alpha, m_input->desc(), m_input->data(),
			&alpha, m_output->desc(), m_output->data());
	}

private:
	const Tensor4* m_input;
	Tensor4* m_output;
};

class Average2D
{
public:
	Average2D(const Tensor4* input, Tensor4* output, const Dim2& win, const Dim2& pad, const Dim2& stride)
		: m_input(input), m_output(output)
	{
		cudnnCreatePoolingDescriptor(&m_desc);
		cudnnSetPoolingNdDescriptor(m_desc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_PROPAGATE_NAN, 2, win.dim, pad.dim, stride.dim);
	}

	~Average2D()
	{
		cudnnDestroyPoolingDescriptor(m_desc);
	}

	void Run() const
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		cudnnPoolingForward(HandleCUDNN::handle(), m_desc,
			&alpha, m_input->desc(), m_input->data(),
			&beta, m_output->desc(), m_output->data());
	}

private:
	cudnnPoolingDescriptor_t m_desc;
	const Tensor4* m_input;
	Tensor4* m_output;
};
