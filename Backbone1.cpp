#include "Backbone1.h"
#include "tensors/backbone1_0_weight.hpp"
#include "tensors/backbone1_0_bias.hpp"
#include "tensors/backbone1_2_0_weight.hpp"
#include "tensors/backbone1_2_1_weight.hpp"
#include "tensors/backbone1_2_1_bias.hpp"
#include "tensors/backbone1_3_0_weight.hpp"
#include "tensors/backbone1_3_1_weight.hpp"
#include "tensors/backbone1_3_1_bias.hpp"
#include "tensors/backbone1_4_0_weight.hpp"
#include "tensors/backbone1_4_1_weight.hpp"
#include "tensors/backbone1_4_1_bias.hpp"
#include "tensors/backbone1_5_0_weight.hpp"
#include "tensors/backbone1_5_1_weight.hpp"
#include "tensors/backbone1_5_1_bias.hpp"
#include "tensors/backbone1_6_0_weight.hpp"
#include "tensors/backbone1_6_1_weight.hpp"
#include "tensors/backbone1_6_1_bias.hpp"
#include "tensors/backbone1_7_0_weight.hpp"
#include "tensors/backbone1_7_1_weight.hpp"
#include "tensors/backbone1_7_1_bias.hpp"
#include "tensors/backbone1_8_0_weight.hpp"
#include "tensors/backbone1_8_1_weight.hpp"
#include "tensors/backbone1_8_1_bias.hpp"
#include "tensors/backbone1_9_0_weight.hpp"
#include "tensors/backbone1_9_1_weight.hpp"
#include "tensors/backbone1_9_1_bias.hpp"
#include "tensors/backbone1_10_0_weight.hpp"
#include "tensors/backbone1_10_1_weight.hpp"
#include "tensors/backbone1_10_1_bias.hpp"
#include "tensors/backbone1_11_0_weight.hpp"
#include "tensors/backbone1_11_1_weight.hpp"
#include "tensors/backbone1_11_1_bias.hpp"
#include "tensors/backbone1_12_0_weight.hpp"
#include "tensors/backbone1_12_1_weight.hpp"
#include "tensors/backbone1_12_1_bias.hpp"

/// 1-01
F_Backbone1_0::F_Backbone1_0(const Tensor4* input, Tensor4* output)
	: m_conv(input, &m_weights, &m_bias, output, { 0,0 }, { 2,2 })
	, m_weights({ 24,3,5,5 })
	, m_bias({1,24,1,1})
{
	m_weights.from_host((float*)backbone1_0_weight);
	m_bias.from_host((float*)backbone1_0_bias);
}

void F_Backbone1_0::Run() const
{
	m_conv.Run();
}

/// 1-2-0
F_Backbone1_2_0::F_Backbone1_2_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 24, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 24)
{
	m_weights.from_host((float*)backbone1_2_0_weight);	
}

void F_Backbone1_2_0::Run() const
{
	m_conv.Run();
}

/// 1-2-1

F_Backbone1_2_1::F_Backbone1_2_1(const Tensor4* input, Tensor4* output)	
	: m_weights({ 24,24,1,1 })
	, m_bias({ 1,24,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_2_1_weight);
	m_bias.from_host((float*)backbone1_2_1_bias);
}

void F_Backbone1_2_1::Run() const
{
	m_conv.Run();
}

/// 1-2
F_Backbone1_2::F_Backbone1_2(const Tensor4* input, Tensor4* output)
	: m_t0({ 1,24,64,64 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_add(input, output)
	, m_relu(output)
{

}

void F_Backbone1_2::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1-3-0
F_Backbone1_3_0::F_Backbone1_3_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 24, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 24)
{
	m_weights.from_host((float*)backbone1_3_0_weight);
}


void F_Backbone1_3_0::Run() const
{
	m_conv.Run();
}

/// 1-3-1
F_Backbone1_3_1::F_Backbone1_3_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 28,24,1,1 })
	, m_bias({ 1,28,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_3_1_weight);
	m_bias.from_host((float*)backbone1_3_1_bias);	
}


void F_Backbone1_3_1::Run() const
{
	m_conv.Run();
}

/// 1-3
F_Backbone1_3::F_Backbone1_3(const Tensor4* input, Tensor4* output)
	: m_t0({ 1,24,64,64 })
	, m_t_x_cpad({1,28,64,64})
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,4,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{	

}


void F_Backbone1_3::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1-4-0
F_Backbone1_4_0::F_Backbone1_4_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 28, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 0,0 }, { 2,2 }, { 1,1 }, 28)
{
	m_weights.from_host((float*)backbone1_4_0_weight);
}

void F_Backbone1_4_0::Run() const
{
	m_conv.Run();
}

/// 1-4-1
F_Backbone1_4_1::F_Backbone1_4_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 32,28,1,1 })
	, m_bias({ 1,32,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_4_1_weight);
	m_bias.from_host((float*)backbone1_4_1_bias);
}

void F_Backbone1_4_1::Run() const
{
	m_conv.Run();
}

/// 1-4
F_Backbone1_4::F_Backbone1_4(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 28, 32, 32 })
	, m_t_x_pad({ 1, 28, 66, 66 })
	, m_t_x_pool({ 1, 28, 32, 32 })
	, m_t_x_cpad({ 1, 32, 32, 32 })
	, m_f0(&m_t_x_pad, &m_t0)
	, m_f1(&m_t0, output)
	, m_pad(input, &m_t_x_pad, { 0,0,0,0 }, { 0,0,2,2 })
	, m_pool(input, &m_t_x_pool, { 2,2 }, { 0,0 }, { 2,2 })
	, m_cpad(&m_t_x_pool, &m_t_x_cpad, {0,0,0,0}, {0,4,0,0})
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{

}

void F_Backbone1_4::Run() const
{
	m_pad.Run();
	m_f0.Run();
	m_f1.Run();
	m_pool.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1-5-0
F_Backbone1_5_0::F_Backbone1_5_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 32, 1, 3,3 })
	, m_conv(input, &m_weights, output, {1,1}, {1,1}, {1,1}, 32)
{
	m_weights.from_host((float*)backbone1_5_0_weight);
}

void F_Backbone1_5_0::Run() const
{
	m_conv.Run();
}

/// 1-5-1
F_Backbone1_5_1::F_Backbone1_5_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 36,32,1,1 })
	, m_bias({ 1, 36,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_5_1_weight);
	m_bias.from_host((float*)backbone1_5_1_bias);
}


void F_Backbone1_5_1::Run() const
{
	m_conv.Run();
}

/// 1-5
F_Backbone1_5::F_Backbone1_5(const Tensor4* input, Tensor4* output)
	: m_input(input), m_output(output)
	, m_t0({ 1, 32, 32, 32 })
	, m_t_x_cpad({ 1, 36, 32,32 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,4,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{
	
}

void F_Backbone1_5::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1-6-0
F_Backbone1_6_0::F_Backbone1_6_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 36, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 },36)
{
	m_weights.from_host((float*)backbone1_6_0_weight);
}

void F_Backbone1_6_0::Run() const
{
	m_conv.Run();
}

/// 1-6-1
F_Backbone1_6_1::F_Backbone1_6_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 42,36,1,1 })
	, m_bias({ 1, 42,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_6_1_weight);
	m_bias.from_host((float*)backbone1_6_1_bias);
}


void F_Backbone1_6_1::Run() const
{
	m_conv.Run();
}

/// 1-6
F_Backbone1_6::F_Backbone1_6(const Tensor4* input, Tensor4* output)
	:  m_t0({ 1, 36, 32, 32 })
	, m_t_x_cpad({ 1, 42, 32,32 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,6,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{	
}

void F_Backbone1_6::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}


/// 1-7-0
F_Backbone1_7_0::F_Backbone1_7_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 42, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 0,0 }, { 2,2 }, { 1,1 }, 42)
{
	m_weights.from_host((float*)backbone1_7_0_weight);
}

void F_Backbone1_7_0::Run() const
{
	m_conv.Run();
}


/// 1-7-1
F_Backbone1_7_1::F_Backbone1_7_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 48,42,1,1 })
	, m_bias({ 1,48,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_7_1_weight);
	m_bias.from_host((float*)backbone1_7_1_bias);
}

void F_Backbone1_7_1::Run() const
{
	m_conv.Run();
}

/// 1-7
F_Backbone1_7::F_Backbone1_7(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 42, 16, 16 })
	, m_t_x_pad({ 1, 42, 34, 34 })
	, m_t_x_pool({ 1, 42, 16, 16 })
	, m_t_x_cpad({ 1, 48, 16, 16 })
	, m_f0(&m_t_x_pad, &m_t0)
	, m_f1(&m_t0, output)
	, m_pad(input, &m_t_x_pad, { 0,0,0,0 }, { 0,0,2,2 })
	, m_pool(input, &m_t_x_pool, { 2,2 }, { 0,0 }, { 2,2 })
	, m_cpad(&m_t_x_pool, &m_t_x_cpad, { 0,0,0,0 }, { 0,6,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{

}

void F_Backbone1_7::Run() const
{
	m_pad.Run();
	m_f0.Run();
	m_f1.Run();
	m_pool.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1-8-0
F_Backbone1_8_0::F_Backbone1_8_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 48, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 48)
{
	m_weights.from_host((float*)backbone1_8_0_weight);
}

void F_Backbone1_8_0::Run() const
{
	m_conv.Run();
}


/// 1-8-1
F_Backbone1_8_1::F_Backbone1_8_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 56,48,1,1 })
	, m_bias({ 1,56,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_8_1_weight);
	m_bias.from_host((float*)backbone1_8_1_bias);
}

void F_Backbone1_8_1::Run() const
{
	m_conv.Run();
}

/// 1-8
F_Backbone1_8::F_Backbone1_8(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 48, 16, 16 })
	, m_t_x_cpad({ 1, 56, 16, 16 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,8,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{
}

void F_Backbone1_8::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1-9-0
F_Backbone1_9_0::F_Backbone1_9_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 56, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 56)
{
	m_weights.from_host((float*)backbone1_9_0_weight);
}

void F_Backbone1_9_0::Run() const
{
	m_conv.Run();
}

/// 1-9-1
F_Backbone1_9_1::F_Backbone1_9_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 64,56,1,1 })
	, m_bias({ 1,64,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_9_1_weight);
	m_bias.from_host((float*)backbone1_9_1_bias);
}

void F_Backbone1_9_1::Run() const
{
	m_conv.Run();
}

/// 1-9
F_Backbone1_9::F_Backbone1_9(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 56, 16, 16 })
	, m_t_x_cpad({ 1, 64, 16, 16 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,8,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{
}

void F_Backbone1_9::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1-10-0
F_Backbone1_10_0::F_Backbone1_10_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 64, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 64)
{
	m_weights.from_host((float*)backbone1_10_0_weight);
}

void F_Backbone1_10_0::Run() const
{
	m_conv.Run();
}

/// 1-10-1
F_Backbone1_10_1::F_Backbone1_10_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 72,64,1,1 })
	, m_bias({ 1,72,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_10_1_weight);
	m_bias.from_host((float*)backbone1_10_1_bias);
}

void F_Backbone1_10_1::Run() const
{
	m_conv.Run();
}

/// 1-10
F_Backbone1_10::F_Backbone1_10(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 64, 16, 16 })
	, m_t_x_cpad({ 1, 72, 16, 16 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,8,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{
}

void F_Backbone1_10::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}


/// 1-11-0
F_Backbone1_11_0::F_Backbone1_11_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 72, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 72)
{
	m_weights.from_host((float*)backbone1_11_0_weight);
}

void F_Backbone1_11_0::Run() const
{
	m_conv.Run();
}

/// 1-11-1
F_Backbone1_11_1::F_Backbone1_11_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 80,72,1,1 })
	, m_bias({ 1,80,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_11_1_weight);
	m_bias.from_host((float*)backbone1_11_1_bias);
}

void F_Backbone1_11_1::Run() const
{
	m_conv.Run();
}

/// 1-11
F_Backbone1_11::F_Backbone1_11(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 72, 16, 16 })
	, m_t_x_cpad({ 1, 80, 16, 16 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,8,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{
}

void F_Backbone1_11::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}


/// 1-12-0
F_Backbone1_12_0::F_Backbone1_12_0(const Tensor4* input, Tensor4* output)
	: m_weights({ 80, 1, 3,3 })
	, m_conv(input, &m_weights, output, { 1,1 }, { 1,1 }, { 1,1 }, 80)
{
	m_weights.from_host((float*)backbone1_12_0_weight);
}

void F_Backbone1_12_0::Run() const
{
	m_conv.Run();
}

/// 1-12-1
F_Backbone1_12_1::F_Backbone1_12_1(const Tensor4* input, Tensor4* output)
	: m_weights({ 88,80,1,1 })
	, m_bias({ 1,88,1,1 })
	, m_conv(input, &m_weights, &m_bias, output)
{
	m_weights.from_host((float*)backbone1_12_1_weight);
	m_bias.from_host((float*)backbone1_12_1_bias);
}

void F_Backbone1_12_1::Run() const
{
	m_conv.Run();
}

/// 1-12
F_Backbone1_12::F_Backbone1_12(const Tensor4* input, Tensor4* output)
	: m_t0({ 1, 80, 16, 16 })
	, m_t_x_cpad({ 1, 88, 16, 16 })
	, m_f0(input, &m_t0)
	, m_f1(&m_t0, output)
	, m_cpad(input, &m_t_x_cpad, { 0,0,0,0 }, { 0,8,0,0 })
	, m_add(&m_t_x_cpad, output)
	, m_relu(output)
{
}

void F_Backbone1_12::Run() const
{
	m_f0.Run();
	m_f1.Run();
	m_cpad.Run();
	m_add.Run();
	m_relu.Run();
}

/// 1
F_Backbone1::F_Backbone1(const Tensor4* input, Tensor4* output)
	:m_t0({ 1,24,64,64 }), m_t2({ 1,24,64,64 }), m_t3({ 1,28,64,64 })
	, m_t4({ 1,32,32,32 }), m_t5({ 1,36,32,32 }), m_t6({ 1,42,32,32 })
	, m_t7({ 1,48,16,16 }), m_t8({ 1,56,16,16 }), m_t9({ 1,64,16,16 })
	, m_t10({ 1,72,16,16 }), m_t11({ 1,80,16,16 })
	, m_f0(input, &m_t0), m_f2(&m_t0, &m_t2), m_f3(&m_t2, &m_t3)
	, m_f4(&m_t3, &m_t4), m_f5(&m_t4, &m_t5), m_f6(&m_t5, &m_t6)
	, m_f7(&m_t6, &m_t7), m_f8(&m_t7, &m_t8), m_f9(&m_t8, &m_t9)
	, m_f10(&m_t9, &m_t10), m_f11(&m_t10, &m_t11), m_f12(&m_t11, output)
{


}

void F_Backbone1::Run() const
{
	m_f0.Run(); m_f2.Run(); m_f3.Run(); m_f4.Run(); m_f5.Run(); m_f6.Run();
	m_f7.Run(), m_f8.Run(); m_f9.Run(); m_f10.Run(); m_f11.Run(); m_f12.Run();
}

