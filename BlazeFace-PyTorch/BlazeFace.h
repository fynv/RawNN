#pragma once

#include "Backbone1.h"
#include "Backbone2.h"

#include <vector>

struct Detection
{
	float coordinates[16];
	float score;
};

class F_BlazeFace
{
public:
	F_BlazeFace();
	void Run(const unsigned char* rgb128);
	const std::vector<Detection>& GetResults()
	{
		return m_output_detections;
	}

private:
	std::vector<float> m_input_nchw;
	std::vector<Detection> m_output_detections;

	Tensor4 m_t_input;
	Tensor4 m_t_bb1;
	Tensor4 m_t_bb2;
	F_Backbone1 m_f_bb1;
	F_Backbone2 m_f_bb2;

	Tensor4 m_t_c1;
	FilterWeights4 m_w_classifier_8;
	Tensor4 m_b_classifier_8;
	Convolution2DBias m_classifier_8;

	Tensor4 m_t_c2;
	FilterWeights4 m_w_classifier_16;
	Tensor4 m_b_classifier_16;
	Convolution2DBias m_classifier_16;

	Tensor4 m_t_r1;
	FilterWeights4 m_w_regressor_8;
	Tensor4 m_b_regressor_8;
	Convolution2DBias m_regressor_8;

	Tensor4 m_t_r2;
	FilterWeights4 m_w_regressor_16;
	Tensor4 m_b_regressor_16;
	Convolution2DBias m_regressor_16;


};