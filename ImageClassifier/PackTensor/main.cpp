#include <stdio.h>
#include <vector>
#include <string>

const std::string root = "../tensors/";

void pack(const char* fn_data, const char* fn_hpp, const char* name_arr)
{
	FILE* fp_hpp = fopen((root+fn_hpp).c_str(), "w");

	FILE* fp_data = fopen((root + fn_data).c_str(), "rb");
	fseek(fp_data, 0, SEEK_END);
	size_t size = (size_t)ftell(fp_data);
	fseek(fp_data, 0, SEEK_SET);
	std::vector<char> buf((size + 3) & (~3));
	fread(buf.data(), 1, size, fp_data);
	fclose(fp_data);

	fprintf(fp_hpp, "const static size_t %s_size = %llu;\n", name_arr, size);

	size_t num_dwords = buf.size() >> 2;
	unsigned* dwords = (unsigned*)buf.data();

	fprintf(fp_hpp, "static unsigned %s[]={\n", name_arr);
	for (size_t j = 0; j < num_dwords; j++)
	{
		fprintf(fp_hpp, "0x%x,", dwords[j]);
		if (j % 10 == 9)
			fputs("\n", fp_hpp);
	}
	fputs("};\n\n", fp_hpp);

	fclose(fp_hpp);

}

int main()
{
	pack("layer3_weight", "layer3_weight.hpp", "layer3_weight");
	pack("layer3_bias", "layer3_bias.hpp", "layer3_bias");
	pack("layer4_scale", "layer4_scale.hpp", "layer4_scale");
	pack("layer4_bias", "layer4_bias.hpp", "layer4_bias");
	pack("layer4_mean", "layer4_mean.hpp", "layer4_mean");
	pack("layer4_variance", "layer4_variance.hpp", "layer4_variance");
	pack("layer6_weight", "layer6_weight.hpp", "layer6_weight");
	pack("layer6_bias", "layer6_bias.hpp", "layer6_bias");
	pack("layer7_scale", "layer7_scale.hpp", "layer7_scale");
	pack("layer7_bias", "layer7_bias.hpp", "layer7_bias");
	pack("layer7_mean", "layer7_mean.hpp", "layer7_mean");
	pack("layer7_variance", "layer7_variance.hpp", "layer7_variance");
	pack("layer10_weight0", "layer10_weight0.hpp", "layer10_weight0");
	pack("layer10_weight1", "layer10_weight1.hpp", "layer10_weight1");
	pack("layer10_bias", "layer10_bias.hpp", "layer10_bias");
	pack("layer11_scale", "layer11_scale.hpp", "layer11_scale");
	pack("layer11_bias", "layer11_bias.hpp", "layer11_bias");
	pack("layer11_mean", "layer11_mean.hpp", "layer11_mean");
	pack("layer11_variance", "layer11_variance.hpp", "layer11_variance");
	pack("layer13_weight0", "layer13_weight0.hpp", "layer13_weight0");
	pack("layer13_weight1", "layer13_weight1.hpp", "layer13_weight1");
	pack("layer13_bias", "layer13_bias.hpp", "layer13_bias");
	pack("layer14_scale", "layer14_scale.hpp", "layer14_scale");
	pack("layer14_bias", "layer14_bias.hpp", "layer14_bias");
	pack("layer14_mean", "layer14_mean.hpp", "layer14_mean");
	pack("layer14_variance", "layer14_variance.hpp", "layer14_variance");
	pack("layer16_weight", "layer16_weight.hpp", "layer16_weight");
	pack("layer16_bias", "layer16_bias.hpp", "layer16_bias");
	pack("layer19_weight0", "layer19_weight0.hpp", "layer19_weight0");
	pack("layer19_weight1", "layer19_weight1.hpp", "layer19_weight1");
	pack("layer19_bias", "layer19_bias.hpp", "layer19_bias");
	pack("layer20_scale", "layer20_scale.hpp", "layer20_scale");
	pack("layer20_bias", "layer20_bias.hpp", "layer20_bias");
	pack("layer20_mean", "layer20_mean.hpp", "layer20_mean");
	pack("layer20_variance", "layer20_variance.hpp", "layer20_variance");
	pack("layer22_weight0", "layer22_weight0.hpp", "layer22_weight0");
	pack("layer22_weight1", "layer22_weight1.hpp", "layer22_weight1");
	pack("layer22_bias", "layer22_bias.hpp", "layer22_bias");
	pack("layer23_scale", "layer23_scale.hpp", "layer23_scale");
	pack("layer23_bias", "layer23_bias.hpp", "layer23_bias");
	pack("layer23_mean", "layer23_mean.hpp", "layer23_mean");
	pack("layer23_variance", "layer23_variance.hpp", "layer23_variance");
	pack("layer25_weight", "layer25_weight.hpp", "layer25_weight");
	pack("layer25_bias", "layer25_bias.hpp", "layer25_bias");
	pack("layer28_weight0", "layer28_weight0.hpp", "layer28_weight0");
	pack("layer28_weight1", "layer28_weight1.hpp", "layer28_weight1");
	pack("layer28_bias", "layer28_bias.hpp", "layer28_bias");
	pack("layer29_scale", "layer29_scale.hpp", "layer29_scale");
	pack("layer29_bias", "layer29_bias.hpp", "layer29_bias");
	pack("layer29_mean", "layer29_mean.hpp", "layer29_mean");
	pack("layer29_variance", "layer29_variance.hpp", "layer29_variance");
	pack("layer31_weight0", "layer31_weight0.hpp", "layer31_weight0");
	pack("layer31_weight1", "layer31_weight1.hpp", "layer31_weight1");
	pack("layer31_bias", "layer31_bias.hpp", "layer31_bias");
	pack("layer32_scale", "layer32_scale.hpp", "layer32_scale");
	pack("layer32_bias", "layer32_bias.hpp", "layer32_bias");
	pack("layer32_mean", "layer32_mean.hpp", "layer32_mean");
	pack("layer32_variance", "layer32_variance.hpp", "layer32_variance");
	pack("layer34_weight", "layer34_weight.hpp", "layer34_weight");
	pack("layer34_bias", "layer34_bias.hpp", "layer34_bias");
	pack("layer37_weight0", "layer37_weight0.hpp", "layer37_weight0");
	pack("layer37_weight1", "layer37_weight1.hpp", "layer37_weight1");
	pack("layer37_bias", "layer37_bias.hpp", "layer37_bias");
	pack("layer38_scale", "layer38_scale.hpp", "layer38_scale");
	pack("layer38_bias", "layer38_bias.hpp", "layer38_bias");
	pack("layer38_mean", "layer38_mean.hpp", "layer38_mean");
	pack("layer38_variance", "layer38_variance.hpp", "layer38_variance");
	pack("layer40_weight0", "layer40_weight0.hpp", "layer40_weight0");
	pack("layer40_weight1", "layer40_weight1.hpp", "layer40_weight1");
	pack("layer40_bias", "layer40_bias.hpp", "layer40_bias");
	pack("layer41_scale", "layer41_scale.hpp", "layer41_scale");
	pack("layer41_bias", "layer41_bias.hpp", "layer41_bias");
	pack("layer41_mean", "layer41_mean.hpp", "layer41_mean");
	pack("layer41_variance", "layer41_variance.hpp", "layer41_variance");
	pack("layer43_weight", "layer43_weight.hpp", "layer43_weight");
	pack("layer43_bias", "layer43_bias.hpp", "layer43_bias");
	pack("layer45_weight0", "layer45_weight0.hpp", "layer45_weight0");
	pack("layer45_weight1", "layer45_weight1.hpp", "layer45_weight1");
	pack("layer45_bias", "layer45_bias.hpp", "layer45_bias");
	pack("layer46_scale", "layer46_scale.hpp", "layer46_scale");
	pack("layer46_bias", "layer46_bias.hpp", "layer46_bias");
	pack("layer46_mean", "layer46_mean.hpp", "layer46_mean");
	pack("layer46_variance", "layer46_variance.hpp", "layer46_variance");
	pack("layer50_weight", "layer50_weight.hpp", "layer50_weight");
	pack("layer50_bias", "layer50_bias.hpp", "layer50_bias");
	return 0;
}
