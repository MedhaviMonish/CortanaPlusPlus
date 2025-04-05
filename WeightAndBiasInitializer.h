#include <vector>
using namespace std;
class WeightAndBiasInitializer
{
	public:
		void initWeightZeroes(vector<float> weight, vector<int> shape) {
			int length = 1;
			for (int i = 0; i < shape.size(); ++i) {
				length *= shape[i];
			}

			for (int i = 0; i < length; ++i) {
				weight[i] = 0;
			}
		}

		void initWeightOnes(vector<float> weight, vector<int> shape) {
			int length = 1;
			for (int i = 0; i < shape.size(); ++i) {
				length *= shape[i];
			}

			for (int i = 0; i < length; ++i) {
				weight[i] = 1.0;
			}
		}
};

