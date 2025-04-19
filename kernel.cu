#include "Tensor.h"
#include "WeightBiasInitializer.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {2, 6};
    auto pair = WeightBiasInitializer<float>::initWeightRandom(shape_input, 2);
    Tensor<float> W = pair.first;
    Tensor<float> B = pair.second;
    cout << W.print();
    cout << B.print();

    return 0;
}
