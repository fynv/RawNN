#include "Backbone2.h"
#include "tensors/backbone2_0_0_weight.hpp"
#include "tensors/backbone2_0_1_weight.hpp"
#include "tensors/backbone2_0_1_bias.hpp"
#include "tensors/backbone2_1_0_weight.hpp"
#include "tensors/backbone2_1_1_weight.hpp"
#include "tensors/backbone2_1_1_bias.hpp"
#include "tensors/backbone2_2_0_weight.hpp"
#include "tensors/backbone2_2_1_weight.hpp"
#include "tensors/backbone2_2_1_bias.hpp"
#include "tensors/backbone2_3_0_weight.hpp"
#include "tensors/backbone2_3_1_weight.hpp"
#include "tensors/backbone2_3_1_bias.hpp"
#include "tensors/backbone2_4_0_weight.hpp"
#include "tensors/backbone2_4_1_weight.hpp"
#include "tensors/backbone2_4_1_bias.hpp"

/// 2-0-0
F_Backbone2_0_0::F_Backbone2_0_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 88, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 0,0 }, { 2,2 }, { 1,1 }, 88)
{
	m_weights.from_host((float*)backbone2_0_0_weight);
}

void F_Backbone2_0_0::Run() const
{
	m_conv.Run();
}

/// 2-0-1
F_Backbone2_0_1::F_Backbone2_0_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 96,88,1,1 })
	, m_bias({ 1,96,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone2_0_1_weight);
	m_bias.from_host((float*)backbone2_0_1_bias);
}

void F_Backbone2_0_1::Run() const
{
	m_conv.Run();
}

/// 2-0
F_Backbone2_0::F_Backbone2_0(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 88, 8, 8 })
	, m_t_x_pad({ 1, 88, 18, 18 })
	, m_t_x_pool({ 1, 88, 8, 8 })
	, m_t_x_cpad({ 1, 96, 8, 8 })
	, m_f0(&m_t_x_pad, &m_t0)
	, m_f1(&m_t0, output)
	, m_pad(input, &m_t_x_pad, { 0,0,0,0 }, { 0,0,2,2 })
	, m_pool(input, &m_t_x_pool, { 2,2 }, { 0,0 }, { 2,2 })
	, m_cpad(&m_t_x_pool, &m_t_x_cpad, { 0,0,0,0 }, { 0,8,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{

}

void F_Backbone2_0::Run() const
{
	m_pad.Run();
	m_f0.Run();
	m_f1.Run();
	m_pool.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 2-1-0
F_Backbone2_1_0::F_Backbone2_1_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 96, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 96)
{
	m_weights.from_host((float*)backbone2_1_0_weight);
}

void F_Backbone2_1_0::Run() const
{
	m_conv.Run();
}

/// 2-1-1
F_Backbone2_1_1::F_Backbone2_1_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 96,96,1,1 })
	, m_bias({ 1,96,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone2_1_1_weight);
	m_bias.from_host((float*)backbone2_1_1_bias);
}

void F_Backbone2_1_1::Run() const
{
	m_conv.Run();
}

/// 2-1
F_Backbone2_1::F_Backbone2_1(const Tensor4* input, Tensor4* output)
	: m_t0({ 1,96,8,8 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_add(input, output)
	, m_relu(output)
{

}

void F_Backbone2_1::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_add.Run();
	m_relu.Run();
}

/// 2-2-0
F_Backbone2_2_0::F_Backbone2_2_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 96, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 96)
{
	m_weights.from_host((float*)backbone2_2_0_weight);
}

void F_Backbone2_2_0::Run() const
{
	m_conv.Run();
}

/// 2-2-1
F_Backbone2_2_1::F_Backbone2_2_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 96,96,1,1 })
	, m_bias({ 1,96,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone2_2_1_weight);
	m_bias.from_host((float*)backbone2_2_1_bias);
}

void F_Backbone2_2_1::Run() const
{
	m_conv.Run();
}

/// 2-2
F_Backbone2_2::F_Backbone2_2(const Tensor4* input, Tensor4* output)
	: m_t0({ 1,96,8,8 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_add(input, output)
	, m_relu(output)
{

}

void F_Backbone2_2::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_add.Run();
	m_relu.Run();
}

/// 2-3-0
F_Backbone2_3_0::F_Backbone2_3_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 96, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 96)
{
	m_weights.from_host((float*)backbone2_3_0_weight);
}

void F_Backbone2_3_0::Run() const
{
	m_conv.Run();
}

/// 2-3-1
F_Backbone2_3_1::F_Backbone2_3_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 96,96,1,1 })
	, m_bias({ 1,96,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone2_3_1_weight);
	m_bias.from_host((float*)backbone2_3_1_bias);
}

void F_Backbone2_3_1::Run() const
{
	m_conv.Run();
}

/// 2-3
F_Backbone2_3::F_Backbone2_3(const Tensor4* input, Tensor4* output)
	: m_t0({ 1,96,8,8 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_add(input, output)
	, m_relu(output)
{

}

void F_Backbone2_3::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_add.Run();
	m_relu.Run();
}

/// 2-4-0
F_Backbone2_4_0::F_Backbone2_4_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 96, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 96)
{
	m_weights.from_host((float*)backbone2_4_0_weight);
}

void F_Backbone2_4_0::Run() const
{
	m_conv.Run();
}

/// 2-4-1
F_Backbone2_4_1::F_Backbone2_4_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 96,96,1,1 })
	, m_bias({ 1,96,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone2_4_1_weight);
	m_bias.from_host((float*)backbone2_4_1_bias);
}

void F_Backbone2_4_1::Run() const
{
	m_conv.Run();
}

/// 2-4
F_Backbone2_4::F_Backbone2_4(const Tensor4* input, Tensor4* output)
	: m_t0({ 1,96,8,8 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_add(input, output)
	, m_relu(output)
{

}

void F_Backbone2_4::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_add.Run();
	m_relu.Run();
}

/// 2
F_Backbone2::F_Backbone2(const Tensor4* input, Tensor4* output)
	: m_t0({ 1,96,8,8 }), m_t1({ 1,96,8,8 }), m_t2({ 1,96,8,8 }), m_t3({ 1,96,8,8 })
	, m_f0(input, &m_t0), m_f1(&m_t0, &m_t1), m_f2(&m_t1, &m_t2), m_f3(&m_t2, &m_t3), m_f4(&m_t3, output)
{

}

void F_Backbone2::Run() const
{
	m_f0.Run(); m_f1.Run(); m_f2.Run(); m_f3.Run(); m_f4.Run();
}
