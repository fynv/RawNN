#pragma once

#include "Tensor.inl"
#include "Op.inl"
#include "HandleCUDNN.h"

class Layer3
{
public:
	Layer3(const Tensor4* input, Tensor4* output);
	void Run() const;

private:

	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Tensor4 m_padded;
	Padding m_padding;
	Convolution2DBias m_conv;

	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;
	BatchNormalization m_bn;

	RELU m_relu;

};

class Layer6
{
public:
	Layer6(const Tensor4* input, Tensor4* output);
	void Run() const;

private:

	FilterWeights4 m_weights;
	Tensor4 m_bias;
	Convolution2DBias m_conv;

	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;
	BatchNormalization m_bn;

	RELU m_relu;
};

class Layer9
{
public:
	Layer9(Tensor4* input, Tensor4* output);
	void Run() const;

private:	
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};

class Layer12
{
public:
	Layer12(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};

class Res9
{
public:
	Res9(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t1;
	Tensor4 m_t2;
	Tensor4 m_t3;
	FilterWeights4 m_weights;
	Tensor4 m_bias;
	
	
	Layer9 m_l9;
	Layer12 m_l12;
	Padding m_padding;
	Pooling2D m_pooling;
	Convolution2DBias m_conv;
	AddTensor m_add;

};


class Layer18
{
public:
	Layer18(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};

class Layer21
{
public:
	Layer21(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};

class Res18
{
public:
	Res18(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t1;
	Tensor4 m_t2;
	Tensor4 m_t3;
	FilterWeights4 m_weights;
	Tensor4 m_bias;


	Layer18 m_l18;
	Layer21 m_l21;	
	Pooling2D m_pooling;
	Padding m_padding;
	Convolution2DBias m_conv;
	AddTensor m_add;

};



class Layer27
{
public:
	Layer27(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};


class Layer30
{
public:
	Layer30(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};

class Res27
{
public:
	Res27(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t1;
	Tensor4 m_t2;
	Tensor4 m_t3;
	FilterWeights4 m_weights;
	Tensor4 m_bias;


	Layer27 m_l27;
	Layer30 m_l30;
	Pooling2D m_pooling;
	Padding m_padding;
	Convolution2DBias m_conv;
	AddTensor m_add;

};


class Layer36
{
public:
	Layer36(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};


class Layer39
{
public:
	Layer39(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;

	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;

	RELU m_relu;
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;
	BatchNormalization m_bn;
};


class Res36
{
public:
	Res36(Tensor4* input, Tensor4* output);
	void Run() const;

private:
	Tensor4 m_t0;
	Tensor4 m_t1;
	Tensor4 m_t2;
	Tensor4 m_t3;
	FilterWeights4 m_weights;
	Tensor4 m_bias;

	Layer36 m_l36;
	Layer39 m_l39;
	Padding m_padding;
	Pooling2D m_pooling;
	Convolution2DBias m_conv;
	AddTensor m_add;

};

class Xception
{
public:
	Xception();
	float Run(const unsigned char* rgb128);

private:
	Tensor4 m_t_input;
	Tensor4 m_t_output;

	Tensor4 m_t5;
	Tensor4 m_t8;

	Tensor4 m_t17;
	Tensor4 m_t26;
	Tensor4 m_t35;
	Tensor4 m_t44;
	Tensor4 m_t45;
	Tensor4 m_t47;

	// layer45 
	FilterWeights4 m_weights0;
	FilterWeights4 m_weights1;
	Tensor4 m_bias;

	// layer46
	Tensor4 m_bn_scale, m_bn_bias, m_bn_mean, m_bn_variance;


	Layer3 m_l3;
	Layer6 m_l6;

	Res9 m_r9;
	Res18 m_r18;
	Res27 m_r27;
	Res36 m_r36;

	// layer45
	Convolution2D m_conv0;
	Convolution2DBias m_conv1;

	// layer46
	BatchNormalization m_bn;

	// layer47
	RELU m_relu;

	// layer48
	Average2D m_ave;

};