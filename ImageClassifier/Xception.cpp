#include "Xception.h"
#include "tensors/layer3_weight.hpp"
#include "tensors/layer3_bias.hpp"
#include "tensors/layer4_scale.hpp"
#include "tensors/layer4_bias.hpp"
#include "tensors/layer4_mean.hpp"
#include "tensors/layer4_variance.hpp"
#include "tensors/layer6_weight.hpp"
#include "tensors/layer6_bias.hpp"
#include "tensors/layer7_scale.hpp"
#include "tensors/layer7_bias.hpp"
#include "tensors/layer7_mean.hpp"
#include "tensors/layer7_variance.hpp"
#include "tensors/layer10_weight0.hpp"
#include "tensors/layer10_weight1.hpp"
#include "tensors/layer10_bias.hpp"
#include "tensors/layer11_scale.hpp"
#include "tensors/layer11_bias.hpp"
#include "tensors/layer11_mean.hpp"
#include "tensors/layer11_variance.hpp"
#include "tensors/layer13_weight0.hpp"
#include "tensors/layer13_weight1.hpp"
#include "tensors/layer13_bias.hpp"
#include "tensors/layer14_scale.hpp"
#include "tensors/layer14_bias.hpp"
#include "tensors/layer14_mean.hpp"
#include "tensors/layer14_variance.hpp"
#include "tensors/layer16_weight.hpp"
#include "tensors/layer16_bias.hpp"
#include "tensors/layer19_weight0.hpp"
#include "tensors/layer19_weight1.hpp"
#include "tensors/layer19_bias.hpp"
#include "tensors/layer20_scale.hpp"
#include "tensors/layer20_bias.hpp"
#include "tensors/layer20_mean.hpp"
#include "tensors/layer20_variance.hpp"
#include "tensors/layer22_weight0.hpp"
#include "tensors/layer22_weight1.hpp"
#include "tensors/layer22_bias.hpp"
#include "tensors/layer23_scale.hpp"
#include "tensors/layer23_bias.hpp"
#include "tensors/layer23_mean.hpp"
#include "tensors/layer23_variance.hpp"
#include "tensors/layer25_weight.hpp"
#include "tensors/layer25_bias.hpp"
#include "tensors/layer28_weight0.hpp"
#include "tensors/layer28_weight1.hpp"
#include "tensors/layer28_bias.hpp"
#include "tensors/layer29_scale.hpp"
#include "tensors/layer29_bias.hpp"
#include "tensors/layer29_mean.hpp"
#include "tensors/layer29_variance.hpp"
#include "tensors/layer31_weight0.hpp"
#include "tensors/layer31_weight1.hpp"
#include "tensors/layer31_bias.hpp"
#include "tensors/layer32_scale.hpp"
#include "tensors/layer32_bias.hpp"
#include "tensors/layer32_mean.hpp"
#include "tensors/layer32_variance.hpp"
#include "tensors/layer34_weight.hpp"
#include "tensors/layer34_bias.hpp"
#include "tensors/layer37_weight0.hpp"
#include "tensors/layer37_weight1.hpp"
#include "tensors/layer37_bias.hpp"
#include "tensors/layer38_scale.hpp"
#include "tensors/layer38_bias.hpp"
#include "tensors/layer38_mean.hpp"
#include "tensors/layer38_variance.hpp"
#include "tensors/layer40_weight0.hpp"
#include "tensors/layer40_weight1.hpp"
#include "tensors/layer40_bias.hpp"
#include "tensors/layer41_scale.hpp"
#include "tensors/layer41_bias.hpp"
#include "tensors/layer41_mean.hpp"
#include "tensors/layer41_variance.hpp"
#include "tensors/layer43_weight.hpp"
#include "tensors/layer43_bias.hpp"
#include "tensors/layer45_weight0.hpp"
#include "tensors/layer45_weight1.hpp"
#include "tensors/layer45_bias.hpp"
#include "tensors/layer46_scale.hpp"
#include "tensors/layer46_bias.hpp"
#include "tensors/layer46_mean.hpp"
#include "tensors/layer46_variance.hpp"
#include "tensors/layer50_weight.hpp"
#include "tensors/layer50_bias.hpp"

/// Layer3 ~ Layer5
Layer3::Layer3(const Tensor4* input, Tensor4* output)
	: m_weights({ 32,3,3,3 })
	, m_bias({ 1,1,1,32 })
	, m_padded({1,181,181,3})
	, m_padding(input, &m_padded, {0,0,0,0}, {0,1,1,0})
	, m_conv(&m_padded, &m_weights, &m_bias, output, { 0,0 }, { 2,2 })
	, m_bn_scale({ 1,1,1,32 })
	, m_bn_bias({ 1,1,1,32 })
	, m_bn_mean({ 1,1,1,32 })
	, m_bn_variance({ 1,1,1,32 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
	, m_relu(output)

{
	m_weights.from_host((float*)layer3_weight);
	m_bias.from_host((float*)layer3_bias);

	m_bn_scale.from_host((float*)layer4_scale);
	m_bn_bias.from_host((float*)layer4_bias);
	m_bn_mean.from_host((float*)layer4_mean);
	m_bn_variance.from_host((float*)layer4_variance);
}

void Layer3::Run() const
{
	m_padding.Run();
	m_conv.Run();
	m_bn.Run();
	m_relu.Run();
}

/// Layer6 ~ 8
Layer6::Layer6(const Tensor4* input, Tensor4* output)
	: m_weights({ 64,3,3,32 })
	, m_bias({ 1,1,1,64 })
	, m_conv(input, &m_weights, &m_bias, output, { 1,1 })
	, m_bn_scale({ 1,1,1,64 })
	, m_bn_bias({ 1,1,1,64 })
	, m_bn_mean({ 1,1,1,64 })
	, m_bn_variance({ 1,1,1,64 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
	, m_relu(output)
{
	m_weights.from_host((float*)layer6_weight);
	m_bias.from_host((float*)layer6_bias);

	m_bn_scale.from_host((float*)layer7_scale);
	m_bn_bias.from_host((float*)layer7_bias);
	m_bn_mean.from_host((float*)layer7_mean);
	m_bn_variance.from_host((float*)layer7_variance);
}


void Layer6::Run() const
{
	m_conv.Run();	
	m_bn.Run();
	m_relu.Run();
}

/// Layer9 ~ 11
Layer9::Layer9(Tensor4* input, Tensor4* output)
	: m_t0({ 1,90,90,64 })
	, m_weights0({64,3,3,1})
	, m_weights1({128,1,1,64})
	, m_bias({ 1,1,1,128 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 64)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,128 })
	, m_bn_bias({ 1,1,1,128 })
	, m_bn_mean({ 1,1,1,128 })
	, m_bn_variance({ 1,1,1,128 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer10_weight0);
	m_weights1.from_host((float*)layer10_weight1);
	m_bias.from_host((float*)layer10_bias);
	m_bn_scale.from_host((float*)layer11_scale);
	m_bn_bias.from_host((float*)layer11_bias);
	m_bn_mean.from_host((float*)layer11_mean);
	m_bn_variance.from_host((float*)layer11_variance);
}

void Layer9::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}

/// Layer12 ~ 14
Layer12::Layer12(Tensor4* input, Tensor4* output)
	: m_t0({ 1,90,90, 128 })
	, m_weights0({ 128,3,3,1 })
	, m_weights1({ 128,1,1, 128 })
	, m_bias({ 1,1,1,128 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 128)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,128 })
	, m_bn_bias({ 1,1,1,128 })
	, m_bn_mean({ 1,1,1,128 })
	, m_bn_variance({ 1,1,1,128 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer13_weight0);
	m_weights1.from_host((float*)layer13_weight1);
	m_bias.from_host((float*)layer13_bias);
	m_bn_scale.from_host((float*)layer14_scale);
	m_bn_bias.from_host((float*)layer14_bias);
	m_bn_mean.from_host((float*)layer14_mean);
	m_bn_variance.from_host((float*)layer14_variance);
}

void Layer12::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}

/// Layer9 ~ 17
Res9::Res9(Tensor4* input, Tensor4* output)
	: m_t0({ 1,90,90,128 })
	, m_t1({ 1,90,90,128 })
	, m_t2({ 1,46,46,128 }) // over-padded
	, m_t3({ 1,45,45,128 })
	, m_weights({128, 1, 1, 64})
	, m_bias({ 1,1,1,128 })
	, m_l9(input, &m_t0)
	, m_l12(&m_t0, &m_t1)
	, m_pooling(&m_t1, &m_t2, { 3,3 }, { 2,2 }, { 2,2 })
	, m_padding(&m_t2, output, { 0,-1,-1,0 }, { 0,0,0,0 }) // cropping
	, m_conv(input, &m_weights, &m_bias, &m_t3, { 0,0 }, { 2,2 })
	, m_add(&m_t3, output)
{
	m_weights.from_host((float*)layer16_weight);
	m_bias.from_host((float*)layer16_bias);
}

void Res9::Run() const
{
	m_conv.Run();

	m_l9.Run();
	m_l12.Run();
	m_pooling.Run();
	m_padding.Run();	
	
	m_add.Run();	

}


/// Layer18 ~ 20
Layer18::Layer18(Tensor4* input, Tensor4* output)
	: m_t0({ 1,45,45,128 })
	, m_weights0({ 128,3,3,1 })
	, m_weights1({ 256,1,1,128 })
	, m_bias({ 1,1,1,256 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 128)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,256 })
	, m_bn_bias({ 1,1,1,256 })
	, m_bn_mean({ 1,1,1,256 })
	, m_bn_variance({ 1,1,1,256 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer19_weight0);
	m_weights1.from_host((float*)layer19_weight1);
	m_bias.from_host((float*)layer19_bias);
	m_bn_scale.from_host((float*)layer20_scale);
	m_bn_bias.from_host((float*)layer20_bias);
	m_bn_mean.from_host((float*)layer20_mean);
	m_bn_variance.from_host((float*)layer20_variance);
}

void Layer18::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}

/// Layer21 ~ 23
Layer21::Layer21(Tensor4* input, Tensor4* output)
	: m_t0({ 1,45,45, 256 })
	, m_weights0({ 256,3,3,1 })
	, m_weights1({ 256,1,1, 256 })
	, m_bias({ 1,1,1,256 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 256)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,256 })
	, m_bn_bias({ 1,1,1,256 })
	, m_bn_mean({ 1,1,1,256 })
	, m_bn_variance({ 1,1,1,256 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer22_weight0);
	m_weights1.from_host((float*)layer22_weight1);
	m_bias.from_host((float*)layer22_bias);
	m_bn_scale.from_host((float*)layer23_scale);
	m_bn_bias.from_host((float*)layer23_bias);
	m_bn_mean.from_host((float*)layer23_mean);
	m_bn_variance.from_host((float*)layer23_variance);
}

void Layer21::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}

/// Layer18 ~ 26
Res18::Res18(Tensor4* input, Tensor4* output)
	: m_t0({ 1,45,45,256 })
	, m_t1({ 1,45,45,256 })
	, m_t2({ 1,46,46,128})
	, m_t3({ 1,23,23,256 })
	, m_weights({ 256, 1, 1, 128 })
	, m_bias({ 1,1,1,256 })
	, m_l18(input, &m_t0)
	, m_l21(&m_t0, &m_t1)
	, m_pooling(&m_t1, output, { 3,3 }, { 1,1 }, { 2,2 })
	, m_padding(input, &m_t2, { 0,0,0,0 }, { 0,1,1,0 })
	, m_conv(&m_t2, &m_weights, &m_bias, &m_t3, { 0,0 }, { 2,2 })
	, m_add(&m_t3, output)
{
	m_weights.from_host((float*)layer25_weight);
	m_bias.from_host((float*)layer25_bias);
}

void Res18::Run() const
{
	m_padding.Run();
	m_conv.Run();

	m_l18.Run();
	m_l21.Run();
	m_pooling.Run();

	m_add.Run();

}


/// Layer27 ~ 29
Layer27::Layer27(Tensor4* input, Tensor4* output)
	: m_t0({ 1,23,23,256 })
	, m_weights0({ 256,3,3,1 })
	, m_weights1({ 512,1,1,256 })
	, m_bias({ 1,1,1,512 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 256)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,512 })
	, m_bn_bias({ 1,1,1,512 })
	, m_bn_mean({ 1,1,1,512 })
	, m_bn_variance({ 1,1,1,512 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer28_weight0);
	m_weights1.from_host((float*)layer28_weight1);
	m_bias.from_host((float*)layer28_bias);
	m_bn_scale.from_host((float*)layer29_scale);
	m_bn_bias.from_host((float*)layer29_bias);
	m_bn_mean.from_host((float*)layer29_mean);
	m_bn_variance.from_host((float*)layer29_variance);
}

void Layer27::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}

/// Layer30 ~ 32
Layer30::Layer30(Tensor4* input, Tensor4* output)
	: m_t0({ 1,23,23, 512 })
	, m_weights0({ 512,3,3,1 })
	, m_weights1({ 512,1,1, 512 })
	, m_bias({ 1,1,1,512 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 512)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,512 })
	, m_bn_bias({ 1,1,1,512 })
	, m_bn_mean({ 1,1,1,512 })
	, m_bn_variance({ 1,1,1,512 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer31_weight0);
	m_weights1.from_host((float*)layer31_weight1);
	m_bias.from_host((float*)layer31_bias);
	m_bn_scale.from_host((float*)layer32_scale);
	m_bn_bias.from_host((float*)layer32_bias);
	m_bn_mean.from_host((float*)layer32_mean);
	m_bn_variance.from_host((float*)layer32_variance);
}

void Layer30::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}

/// Layer27 ~ 35
Res27::Res27(Tensor4* input, Tensor4* output)
	: m_t0({ 1,23,23,512 })
	, m_t1({ 1,23,23,512 })
	, m_t2({ 1,24,24,256 })
	, m_t3({ 1,12,12,512 })
	, m_weights({ 512, 1, 1, 256 })
	, m_bias({ 1,1,1,512 })
	, m_l27(input, &m_t0)
	, m_l30(&m_t0, &m_t1)
	, m_pooling(&m_t1, output, { 3,3 }, { 1,1 }, { 2,2 })
	, m_padding(input, &m_t2, { 0,0,0,0 }, { 0,1,1,0 })
	, m_conv(&m_t2, &m_weights, &m_bias, &m_t3, { 0,0 }, { 2,2 })
	, m_add(&m_t3, output)
{
	m_weights.from_host((float*)layer34_weight);
	m_bias.from_host((float*)layer34_bias);
}

void Res27::Run() const
{
	m_padding.Run();
	m_conv.Run();

	m_l27.Run();
	m_l30.Run();
	m_pooling.Run();

	m_add.Run();

}

/// Layer36 ~ 38
Layer36::Layer36(Tensor4* input, Tensor4* output)
	: m_t0({ 1,12,12,512 })
	, m_weights0({ 512,3,3,1 })
	, m_weights1({ 728,1,1,512 })
	, m_bias({ 1,1,1,728 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 512)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,728 })
	, m_bn_bias({ 1,1,1,728 })
	, m_bn_mean({ 1,1,1,728 })
	, m_bn_variance({ 1,1,1,728 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer37_weight0);
	m_weights1.from_host((float*)layer37_weight1);
	m_bias.from_host((float*)layer37_bias);
	m_bn_scale.from_host((float*)layer38_scale);
	m_bn_bias.from_host((float*)layer38_bias);
	m_bn_mean.from_host((float*)layer38_mean);
	m_bn_variance.from_host((float*)layer38_variance);
}

void Layer36::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}


/// Layer39 ~ 41
Layer39::Layer39(Tensor4* input, Tensor4* output)
	: m_t0({ 1,12,12, 728 })
	, m_weights0({ 728,3,3,1 })
	, m_weights1({ 728,1,1, 728 })
	, m_bias({ 1,1,1, 728 })
	, m_relu(input)
	, m_conv0(input, &m_weights0, &m_t0, { 1,1 }, { 1,1 }, { 1,1 }, 728)
	, m_conv1(&m_t0, &m_weights1, &m_bias, output)
	, m_bn_scale({ 1,1,1,728 })
	, m_bn_bias({ 1,1,1, 728})
	, m_bn_mean({ 1,1,1, 728 })
	, m_bn_variance({ 1,1,1, 728 })
	, m_bn(output, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, output)
{
	m_weights0.from_host((float*)layer40_weight0);
	m_weights1.from_host((float*)layer40_weight1);
	m_bias.from_host((float*)layer40_bias);
	m_bn_scale.from_host((float*)layer41_scale);
	m_bn_bias.from_host((float*)layer41_bias);
	m_bn_mean.from_host((float*)layer41_mean);
	m_bn_variance.from_host((float*)layer41_variance);
}

void Layer39::Run() const
{
	m_relu.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
}


/// Layer36 ~ 44
Res36::Res36(Tensor4* input, Tensor4* output)
	: m_t0({ 1,12,12,728 })
	, m_t1({ 1,12,12,728 })
	, m_t2({ 1,7,7,728 }) // over-padded
	, m_t3({ 1,6,6,728 })
	, m_weights({ 728, 1, 1, 512 })
	, m_bias({ 1,1,1,728 })
	, m_l36(input, &m_t0)
	, m_l39(&m_t0, &m_t1)
	, m_pooling(&m_t1, &m_t2, { 3,3 }, { 2,2 }, { 2,2 })
	, m_padding(&m_t2, output, { 0,-1,-1,0 }, { 0,0,0,0 }) // cropping
	, m_conv(input, &m_weights, &m_bias, &m_t3, { 0,0 }, { 2,2 })
	, m_add(&m_t3, output)
{
	m_weights.from_host((float*)layer43_weight);
	m_bias.from_host((float*)layer43_bias);
}

void Res36::Run() const
{
	m_conv.Run();

	m_l36.Run();
	m_l39.Run();
	m_pooling.Run();
	m_padding.Run();

	m_add.Run();

}

Xception::Xception()
	: m_t_input({ { 1,180,180,3 } })
	, m_t_output({ { 1,1,1,1024 } })
	, m_t5({ 1,90,90,32 })
	, m_t8({ 1,90,90,64 })
	, m_t17({ 1,45,45,128 })
	, m_t26({ 1,23,23,256 })
	, m_t35({ 1,12,12,512 })
	, m_t44({ 1,6,6,728 })
	, m_t45({ 1,6,6, 728 })
	, m_t47({ 1,6,6,1024 })

	, m_weights0({ 728,3,3,1 })
	, m_weights1({ 1024,1,1,728 })
	, m_bias({ 1,1,1,1024 })
	, m_bn_scale({ 1,1,1,1024 })
	, m_bn_bias({ 1,1,1, 1024 })
	, m_bn_mean({ 1,1,1, 1024 })
	, m_bn_variance({ 1,1,1, 1024 })

	, m_l3(&m_t_input, &m_t5)
	, m_l6(&m_t5, &m_t8)
	, m_r9(&m_t8, &m_t17)
	, m_r18(&m_t17, &m_t26)
	, m_r27(&m_t26, &m_t35)
	, m_r36(&m_t35, &m_t44)

	, m_conv0(&m_t44, &m_weights0, &m_t45, { 1,1 }, { 1,1 }, { 1,1 }, 728)
	, m_conv1(&m_t45, &m_weights1, &m_bias, &m_t47)
	, m_bn(&m_t47, &m_bn_scale, &m_bn_bias, &m_bn_mean, &m_bn_variance, &m_t47)
	, m_relu(&m_t47)
	, m_ave(&m_t47, &m_t_output, {6,6}, {0,0}, {1,1})
{
	m_weights0.from_host((float*)layer45_weight0);
	m_weights1.from_host((float*)layer45_weight1);
	m_bias.from_host((float*)layer45_bias);
	m_bn_scale.from_host((float*)layer46_scale);
	m_bn_bias.from_host((float*)layer46_bias);
	m_bn_mean.from_host((float*)layer46_mean);
	m_bn_variance.from_host((float*)layer46_variance);

}

#include <vector>


inline float sigmoid(float x)
{
	float result;
	result = 1.0f / (1.0f + expf(-x));
	return result;
}

float Xception::Run(const unsigned char* rgb128) 
{
	std::vector<float> input(180 * 180 * 3);
	float k = 1.0f / 255.0f;
	for (size_t i = 0; i < input.size(); i++)
	{
		input[i] = (float)rgb128[i]*k;
	}
	m_t_input.from_host(input.data());

	m_l3.Run();
	m_l6.Run();
	m_r9.Run();
	m_r18.Run();
	m_r27.Run();
	m_r36.Run();
	m_conv0.Run();
	m_conv1.Run();
	m_bn.Run();
	m_relu.Run();
	m_ave.Run();

	std::vector<float> output(1024);
	m_t_output.to_host(output.data());

	const float* l50_weight = (const float*)layer50_weight;
	float l50_bias = *(const float*)layer50_bias;

	float v = 0.0f;
	for (int i = 0; i < 1024; i++)
		v += l50_weight[i] * output[i];
	v += l50_bias;

	v = sigmoid(v);

	return v;
}
