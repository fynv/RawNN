#include "BlazeFace.h"
#include "tensors/classifier_8_weight.hpp"
#include "tensors/classifier_8_bias.hpp"
#include "tensors/classifier_16_weight.hpp"
#include "tensors/classifier_16_bias.hpp"
#include "tensors/regressor_8_weight.hpp"
#include "tensors/regressor_8_bias.hpp"
#include "tensors/regressor_16_weight.hpp"
#include "tensors/regressor_16_bias.hpp"
#include "tensors/anchors.hpp"

#include <math.h>
#include <numeric>
#include <algorithm>

F_BlazeFace::F_BlazeFace()
	: m_input_nchw(3 * 131 * 131, 0.0f)
	, m_t_input({ 1,3,131,131 })
	, m_t_bb1({ 1,88,16,16 })
	, m_t_bb2({ 1,96,8,8 })
	, m_f_bb1(&m_t_input, &m_t_bb1)
	, m_f_bb2(&m_t_bb1, &m_t_bb2)

	, m_t_c1({1,2,16,16})
	, m_w_classifier_8({2,88,1,1})
	, m_b_classifier_8({ 1, 2, 1, 1})
	, m_classifier_8(&m_t_bb1, &m_w_classifier_8, &m_b_classifier_8, &m_t_c1)

	, m_t_c2({ 1,6,8,8 })
	, m_w_classifier_16({ 6, 96,1,1 })
	, m_b_classifier_16({ 1, 6, 1, 1 })
	, m_classifier_16(&m_t_bb2, &m_w_classifier_16, &m_b_classifier_16, &m_t_c2)

	, m_t_r1({1,32,16,16})
	, m_w_regressor_8({32,88,1,1})
	, m_b_regressor_8({ 1, 32, 1, 1 })
	, m_regressor_8(&m_t_bb1, &m_w_regressor_8, &m_b_regressor_8, &m_t_r1)

	, m_t_r2({ 1,96,8,8 })
	, m_w_regressor_16({ 96,96,1,1 })
	, m_b_regressor_16({ 1, 96, 1, 1 })
	, m_regressor_16(&m_t_bb2, &m_w_regressor_16, &m_b_regressor_16, &m_t_r2)

{
	m_w_classifier_8.from_host((float*)classifier_8_weight);
	m_b_classifier_8.from_host((float*)classifier_8_bias);
	m_w_classifier_16.from_host((float*)classifier_16_weight);
	m_b_classifier_16.from_host((float*)classifier_16_bias);
	m_w_regressor_8.from_host((float*)regressor_8_weight);
	m_b_regressor_8.from_host((float*)regressor_8_bias);
	m_w_regressor_16.from_host((float*)regressor_16_weight);
	m_b_regressor_16.from_host((float*)regressor_16_bias);
}


inline float sigmoid(float x)
{
	float result;
	result = 1.0f / (1.0f + expf(-x));
	return result;
}

void F_BlazeFace::Run(const unsigned char* rgb128)
{
	float* r_o = &m_input_nchw[0];
	float* g_o = &m_input_nchw[(size_t)131 * 131];
	float* b_o = &m_input_nchw[(size_t)131 * 131 * 2];
	for (int y=0;y< 128; y++)
		for (int x = 0; x < 128; x++)
		{
			float r = (float)(rgb128[(y * 128 + x) * 3]);
			float g = (float)(rgb128[(y * 128 + x) * 3 + 1]);
			float b = (float)(rgb128[(y * 128 + x) * 3 + 2]);
			r_o[(y+1) * 131 + (x+1)] = r / 127.5f - 1.0f;
			g_o[(y+1) * 131 + (x+1)] = g / 127.5f - 1.0f;
			b_o[(y+1) * 131 + (x+1)] = b / 127.5f - 1.0f;
		}

	m_t_input.from_host(m_input_nchw.data());

	m_f_bb1.Run();
	m_f_bb2.Run();
	m_classifier_8.Run();
	m_classifier_16.Run();
	m_regressor_8.Run();
	m_regressor_16.Run();

	float c1[2 * 16 * 16];
	float c2[6 * 8 * 8];
	float r1[32 * 16 * 16];
	float r2[96 * 8 * 8];
	m_t_c1.to_host(c1);
	m_t_c2.to_host(c2);
	m_t_r1.to_host(r1);
	m_t_r2.to_host(r2);

	float c[896];
	float r[896 * 16];
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 2; j++)
			c[i * 2 + j] = c1[256 * j + i];	
	for (int i = 0; i < 64; i++)
		for (int j = 0; j < 6; j++)
			c[512 + i * 6+ j] = c2[64 * j + i];
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 32; j++)
			r[i * 32 + j] = r1[256 * j + i];
	for (int i = 0; i < 64; i++)
		for (int j = 0; j < 96; j++)
			r[8192 + i * 96 + j] = r2[64 * j + i];

	// tensors_to_detections
	const float* p_anchors = (const float*)anchors; // 896 * 4
	// decode_boxes
	float boxes[896 * 16];
	for (int i = 0; i < 896; i++)
	{
		float x_center = r[i * 16] / 128.0f*p_anchors[i * 4 + 2] + p_anchors[i * 4];
		float y_center = r[i * 16 + 1] / 128.0f*p_anchors[i * 4 + 3] + p_anchors[i * 4 + 1];
		float w = r[i * 16 + 2] / 128.0f* p_anchors[i * 4 + 2];
		float h = r[i * 16 + 3] / 128.0f* p_anchors[i * 4 + 3];
		boxes[i * 16] = y_center - h / 2.0f;
		boxes[i * 16 + 1] = x_center - w / 2.0f;
		boxes[i * 16 + 2] = y_center + h / 2.0f;
		boxes[i * 16 + 3] = x_center + w / 2.0f;

		for (int k = 0; k < 6; k++)
		{
			int offset = 4 + k * 2;
			float keypoint_x = r[i * 16 + offset] / 128.0f * p_anchors[i * 4 + 2] + p_anchors[i * 4];
			float keypoint_y = r[i * 16 + offset + 1] / 128.0f * p_anchors[i * 4 + 3] + p_anchors[i * 4 + 1];
			boxes[i * 16 + offset] = keypoint_x;
			boxes[i * 16 + offset + 1] = keypoint_y;
		}
	}

	const float thresh = 100.0f;
	const float min_score_thresh = 0.75f;

	std::vector<Detection> detections;
		
	for (int i = 0; i < 896; i++)
	{
		float score = c[i];
		if (score < -thresh) score = -thresh;
		if (score > thresh) score = thresh;
		score = sigmoid(score);
		c[i] = score;
		if (score >= min_score_thresh)
		{
			Detection d;
			memcpy(d.coordinates, boxes + 16 * i, sizeof(float) * 16);
			d.score = score;			
			detections.push_back(d);
		}
	}

	// Non-maximum suppression
	const float min_suppression_threshold = 0.3f;

	std::vector<int> remaining(detections.size());
	// arg-sort
	std::iota(remaining.begin(), remaining.end(), 0);
	std::sort(remaining.begin(), remaining.end(),
		[&detections](int left, int right) -> bool {
		// sort indices according to corresponding array element
		return detections[left].score < detections[right].score;
	});

	m_output_detections.clear();

	while (remaining.size() > 0)
	{
		const Detection& detection = detections[remaining[0]];
		
		std::vector<int> overlapping;
		std::vector<int> remaining_next;
		
		overlapping.push_back(remaining[0]);

		float area_a = (detection.coordinates[2] - detection.coordinates[0]) * (detection.coordinates[3] - detection.coordinates[1]);

		for (int i = 1; i < (int)remaining.size(); i++)
		{
			int i_detect = remaining[i];
			const Detection& detection_i = detections[i_detect];

			float max_x = std::min(detection.coordinates[2], detection_i.coordinates[2]);
			float max_y = std::min(detection.coordinates[3], detection_i.coordinates[3]);
			float min_x = std::max(detection.coordinates[0], detection_i.coordinates[0]);
			float min_y = std::max(detection.coordinates[1], detection_i.coordinates[1]);

			float len_x = std::max(0.0f, max_x - min_x);
			float len_y = std::max(0.0f, max_y - min_y);
			float intersect = len_x * len_y;

			float area_b = (detection_i.coordinates[2] - detection_i.coordinates[0]) * (detection_i.coordinates[3] - detection_i.coordinates[1]);
			float u = area_a + area_b - intersect;
			float jaccard = intersect / u;		

			if (jaccard > min_suppression_threshold)
				overlapping.push_back(i_detect);
			else
				remaining_next.push_back(i_detect);
		}
		
		Detection weighted_detection = detection;

		if (overlapping.size() > 1)
		{
			memset(&weighted_detection, 0, sizeof(Detection));
			for (int i = 0; i < (int)overlapping.size(); i++)
			{
				int i_overlap = overlapping[i];
				const Detection& detection_i = detections[i_overlap];
				weighted_detection.score += detection_i.score;
				for (int j = 0; j < 16; j++)
					weighted_detection.coordinates[j] += detection_i.coordinates[j] * detection_i.score;
			}
			for (int j = 0; j < 16; j++)
				weighted_detection.coordinates[j] /= weighted_detection.score;
			weighted_detection.score /= (float)overlapping.size();
		}		

		m_output_detections.push_back(weighted_detection);
		remaining = remaining_next;
	}
	
}


