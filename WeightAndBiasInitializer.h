#pragma once

#include <vector>

class WeightAndBiasInitializer
{
public:
	void initWeightZeroes(std::vector<float>& weight, const std::vector<int>& shape);

	void initWeightOnes(std::vector<float>& weight, const std::vector<int>& shape);
};

