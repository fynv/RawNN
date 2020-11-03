#pragma once

#include "Tensor.inl"
#include "Op.inl"
#include "HandleCUDNN.h"

class F_Backbone2_0_0
{
public:
	F_Backbone2_0_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone2_0_1
{
public:
	F_Backbone2_0_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone2_0
{
public:
	F_Backbone2_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t_x_pad;
	Tensor4 m_t0;
	Tensor4 m_t_x_pool;
	Tensor4 m_t_x_cpad;
	F_Backbone2_0_0 m_f0;
	F_Backbone2_0_1 m_f1;
	Padding m_pad;
	Pooling2D m_pool;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};


class F_Backbone2_1_0
{
public:
	F_Backbone2_1_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone2_1_1
{
public:
	F_Backbone2_1_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone2_1
{
public:
	F_Backbone2_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	F_Backbone2_1_0 m_f0;
	F_Backbone2_1_1 m_f1;
	AddTensor m_add;
	RELU m_relu;
};


class F_Backbone2_2_0
{
public:
	F_Backbone2_2_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone2_2_1
{
public:
	F_Backbone2_2_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone2_2
{
public:
	F_Backbone2_2(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	F_Backbone2_2_0 m_f0;
	F_Backbone2_2_1 m_f1;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone2_3_0
{
public:
	F_Backbone2_3_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone2_3_1
{
public:
	F_Backbone2_3_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone2_3
{
public:
	F_Backbone2_3(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	F_Backbone2_3_0 m_f0;
	F_Backbone2_3_1 m_f1;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone2_4_0
{
public:
	F_Backbone2_4_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone2_4_1
{
public:
	F_Backbone2_4_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone2_4
{
public:
	F_Backbone2_4(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	F_Backbone2_4_0 m_f0;
	F_Backbone2_4_1 m_f1;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone2
{
public:
	F_Backbone2(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t1;
	Tensor4 m_t2;
	Tensor4 m_t3;
	F_Backbone2_0 m_f0;
	F_Backbone2_1 m_f1;
	F_Backbone2_2 m_f2;
	F_Backbone2_3 m_f3;
	F_Backbone2_4 m_f4;
};