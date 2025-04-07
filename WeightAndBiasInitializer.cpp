#include "WeightAndBiasInitializer.h"
#include <vector>
using namespace std;

void WeightAndBiasInitializer :: initWeightZeroes(vector<float>& weight, const vector<int>& shape) {
	int length = 1;
	for (int i = 0; i < shape.size(); ++i) {
		length *= shape[i];
	}

	for (int i = 0; i < length; ++i) {
		weight[i] = 0;
	}
}

void WeightAndBiasInitializer::initWeightOnes(vector<float>& weight, const vector<int>& shape) {
	int length = 1;
	for (int i = 0; i < shape.size(); ++i) {
		length *= shape[i];
	}

	for (int i = 0; i < length; ++i) {
		weight[i] = 1.0;
	}
}

