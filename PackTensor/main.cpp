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
	pack("backbone1_0_weight", "backbone1_0_weight.hpp", "backbone1_0_weight");
	pack("backbone1_0_bias", "backbone1_0_bias.hpp", "backbone1_0_bias");
	pack("backbone1_2_0_weight", "backbone1_2_0_weight.hpp", "backbone1_2_0_weight");
	pack("backbone1_2_1_weight", "backbone1_2_1_weight.hpp", "backbone1_2_1_weight");
	pack("backbone1_2_1_bias", "backbone1_2_1_bias.hpp", "backbone1_2_1_bias");
	pack("backbone1_3_0_weight", "backbone1_3_0_weight.hpp", "backbone1_3_0_weight");
	pack("backbone1_3_1_weight", "backbone1_3_1_weight.hpp", "backbone1_3_1_weight");
	pack("backbone1_3_1_bias", "backbone1_3_1_bias.hpp", "backbone1_3_1_bias");
	pack("backbone1_4_0_weight", "backbone1_4_0_weight.hpp", "backbone1_4_0_weight");
	pack("backbone1_4_1_weight", "backbone1_4_1_weight.hpp", "backbone1_4_1_weight");
	pack("backbone1_4_1_bias", "backbone1_4_1_bias.hpp", "backbone1_4_1_bias");
	pack("backbone1_5_0_weight", "backbone1_5_0_weight.hpp", "backbone1_5_0_weight");
	pack("backbone1_5_1_weight", "backbone1_5_1_weight.hpp", "backbone1_5_1_weight");
	pack("backbone1_5_1_bias", "backbone1_5_1_bias.hpp", "backbone1_5_1_bias");
	pack("backbone1_6_0_weight", "backbone1_6_0_weight.hpp", "backbone1_6_0_weight");
	pack("backbone1_6_1_weight", "backbone1_6_1_weight.hpp", "backbone1_6_1_weight");
	pack("backbone1_6_1_bias", "backbone1_6_1_bias.hpp", "backbone1_6_1_bias");
	pack("backbone1_7_0_weight", "backbone1_7_0_weight.hpp", "backbone1_7_0_weight");
	pack("backbone1_7_1_weight", "backbone1_7_1_weight.hpp", "backbone1_7_1_weight");
	pack("backbone1_7_1_bias", "backbone1_7_1_bias.hpp", "backbone1_7_1_bias");
	pack("backbone1_8_0_weight", "backbone1_8_0_weight.hpp", "backbone1_8_0_weight");
	pack("backbone1_8_1_weight", "backbone1_8_1_weight.hpp", "backbone1_8_1_weight");
	pack("backbone1_8_1_bias", "backbone1_8_1_bias.hpp", "backbone1_8_1_bias");
	pack("backbone1_9_0_weight", "backbone1_9_0_weight.hpp", "backbone1_9_0_weight");
	pack("backbone1_9_1_weight", "backbone1_9_1_weight.hpp", "backbone1_9_1_weight");
	pack("backbone1_9_1_bias", "backbone1_9_1_bias.hpp", "backbone1_9_1_bias");
	pack("backbone1_10_0_weight", "backbone1_10_0_weight.hpp", "backbone1_10_0_weight");
	pack("backbone1_10_1_weight", "backbone1_10_1_weight.hpp", "backbone1_10_1_weight");
	pack("backbone1_10_1_bias", "backbone1_10_1_bias.hpp", "backbone1_10_1_bias");
	pack("backbone1_11_0_weight", "backbone1_11_0_weight.hpp", "backbone1_11_0_weight");
	pack("backbone1_11_1_weight", "backbone1_11_1_weight.hpp", "backbone1_11_1_weight");
	pack("backbone1_11_1_bias", "backbone1_11_1_bias.hpp", "backbone1_11_1_bias");
	pack("backbone1_12_0_weight", "backbone1_12_0_weight.hpp", "backbone1_12_0_weight");
	pack("backbone1_12_1_weight", "backbone1_12_1_weight.hpp", "backbone1_12_1_weight");
	pack("backbone1_12_1_bias", "backbone1_12_1_bias.hpp", "backbone1_12_1_bias");

	pack("backbone2_0_0_weight", "backbone2_0_0_weight.hpp", "backbone2_0_0_weight");
	pack("backbone2_0_1_weight", "backbone2_0_1_weight.hpp", "backbone2_0_1_weight");
	pack("backbone2_0_1_bias", "backbone2_0_1_bias.hpp", "backbone2_0_1_bias");
	pack("backbone2_1_0_weight", "backbone2_1_0_weight.hpp", "backbone2_1_0_weight");
	pack("backbone2_1_1_weight", "backbone2_1_1_weight.hpp", "backbone2_1_1_weight");
	pack("backbone2_1_1_bias", "backbone2_1_1_bias.hpp", "backbone2_1_1_bias");
	pack("backbone2_2_0_weight", "backbone2_2_0_weight.hpp", "backbone2_2_0_weight");
	pack("backbone2_2_1_weight", "backbone2_2_1_weight.hpp", "backbone2_2_1_weight");
	pack("backbone2_2_1_bias", "backbone2_2_1_bias.hpp", "backbone2_2_1_bias");
	pack("backbone2_3_0_weight", "backbone2_3_0_weight.hpp", "backbone2_3_0_weight");
	pack("backbone2_3_1_weight", "backbone2_3_1_weight.hpp", "backbone2_3_1_weight");
	pack("backbone2_3_1_bias", "backbone2_3_1_bias.hpp", "backbone2_3_1_bias");
	pack("backbone2_4_0_weight", "backbone2_4_0_weight.hpp", "backbone2_4_0_weight");
	pack("backbone2_4_1_weight", "backbone2_4_1_weight.hpp", "backbone2_4_1_weight");
	pack("backbone2_4_1_bias", "backbone2_4_1_bias.hpp", "backbone2_4_1_bias");

	pack("classifier_8_weight", "classifier_8_weight.hpp", "classifier_8_weight");
	pack("classifier_8_bias", "classifier_8_bias.hpp", "classifier_8_bias");
	pack("classifier_16_weight", "classifier_16_weight.hpp", "classifier_16_weight");
	pack("classifier_16_bias", "classifier_16_bias.hpp", "classifier_16_bias");
	pack("regressor_8_weight", "regressor_8_weight.hpp", "regressor_8_weight");
	pack("regressor_8_bias", "regressor_8_bias.hpp", "regressor_8_bias");
	pack("regressor_16_weight", "regressor_16_weight.hpp", "regressor_16_weight");
	pack("regressor_16_bias", "regressor_16_bias.hpp", "regressor_16_bias");

	pack("anchors", "anchors.hpp", "anchors");

	return 0;
}
