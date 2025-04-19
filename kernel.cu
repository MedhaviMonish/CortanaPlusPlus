#include "Tensor.h"
#include "WeightBiasInitializer.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {2, 6};
    int data_input[12] = {1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};
    auto pair = WeightBiasInitializer<int>::initWeightOnes(shape_input, 2);
    Tensor<int> W = pair.first;
    Tensor<int> B = pair.second;
    cout << W.print();
    cout << B.print();

    return 0;
}
