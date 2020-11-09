#pragma once

#include "Tensor.inl"
#include "Op.inl"
#include "HandleCUDNN.h"

class F_Backbone1_0
{
public:
	F_Backbone1_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBiasRELU m_conv;
};

class F_Backbone1_2_0
{
public:
	F_Backbone1_2_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_2_1
{
public:
	F_Backbone1_2_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_2
{
public:
	F_Backbone1_2(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	F_Backbone1_2_0 m_f0;
	F_Backbone1_2_1 m_f1;
	AddTensor m_add;
	RELU m_relu;
};


class F_Backbone1_3_0
{
public:
	F_Backbone1_3_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};


class F_Backbone1_3_1
{
public:
	F_Backbone1_3_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;	
};

class F_Backbone1_3
{
public:
	F_Backbone1_3(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_3_0 m_f0;
	F_Backbone1_3_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1_4_0
{
public:
	F_Backbone1_4_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_4_1
{
public:
	F_Backbone1_4_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_4
{
public:
	F_Backbone1_4(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t_x_pad;	
	Tensor4 m_t0;
	Tensor4 m_t_x_pool;
	Tensor4 m_t_x_cpad;
	F_Backbone1_4_0 m_f0;
	F_Backbone1_4_1 m_f1;
	Padding m_pad;
	Pooling2D m_pool;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1_5_0
{
public:
	F_Backbone1_5_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_5_1
{
public:
	F_Backbone1_5_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_5
{
public:
	F_Backbone1_5(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	const Tensor4* m_input;
	Tensor4* m_output;
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_5_0 m_f0;
	F_Backbone1_5_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1_6_0
{
public:
	F_Backbone1_6_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_6_1
{
public:
	F_Backbone1_6_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_6
{
public:
	F_Backbone1_6(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_6_0 m_f0;
	F_Backbone1_6_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};


class F_Backbone1_7_0
{
public:
	F_Backbone1_7_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_7_1
{
public:
	F_Backbone1_7_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_7
{
public:
	F_Backbone1_7(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t_x_pad;
	Tensor4 m_t0;
	Tensor4 m_t_x_pool;
	Tensor4 m_t_x_cpad;
	F_Backbone1_7_0 m_f0;
	F_Backbone1_7_1 m_f1;
	Padding m_pad;
	Pooling2D m_pool;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1_8_0
{
public:
	F_Backbone1_8_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_8_1
{
public:
	F_Backbone1_8_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_8
{
public:
	F_Backbone1_8(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_8_0 m_f0;
	F_Backbone1_8_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1_9_0
{
public:
	F_Backbone1_9_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_9_1
{
public:
	F_Backbone1_9_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_9
{
public:
	F_Backbone1_9(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_9_0 m_f0;
	F_Backbone1_9_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};


class F_Backbone1_10_0
{
public:
	F_Backbone1_10_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_10_1
{
public:
	F_Backbone1_10_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_10
{
public:
	F_Backbone1_10(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_10_0 m_f0;
	F_Backbone1_10_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1_11_0
{
public:
	F_Backbone1_11_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_11_1
{
public:
	F_Backbone1_11_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_11
{
public:
	F_Backbone1_11(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_11_0 m_f0;
	F_Backbone1_11_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1_12_0
{
public:
	F_Backbone1_12_0(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Convolution2D m_conv;
};

class F_Backbone1_12_1
{
public:
	F_Backbone1_12_1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;
};

class F_Backbone1_12
{
public:
	F_Backbone1_12(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t_x_cpad;
	F_Backbone1_12_0 m_f0;
	F_Backbone1_12_1 m_f1;
	Padding m_cpad;
	AddTensor m_add;
	RELU m_relu;
};

class F_Backbone1
{
public:
	F_Backbone1(const Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0, m_t2, m_t3, m_t4, m_t5, m_t6, m_t7, m_t8, m_t9, m_t10, m_t11;
	F_Backbone1_0 m_f0;
	F_Backbone1_2 m_f2;
	F_Backbone1_3 m_f3;
	F_Backbone1_4 m_f4;
	F_Backbone1_5 m_f5;
	F_Backbone1_6 m_f6;
	F_Backbone1_7 m_f7;
	F_Backbone1_8 m_f8;
	F_Backbone1_9 m_f9;
	F_Backbone1_10 m_f10;
	F_Backbone1_11 m_f11;
	F_Backbone1_12 m_f12;

};