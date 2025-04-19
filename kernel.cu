#include "DenseLayer.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {10, 6};
    int shape_weight[] = {2, 6};
    Tensor<float> input = Tensor<float>::getOnes(shape_input, 2);
    input = input + 3;
    cout << input.print() << endl;
    DenseLayer<float> layer = DenseLayer<float>(shape_weight, 2, Initialization::ONES);
    cout << layer.weights.print() << endl;
    cout << layer.bias.print() << endl;
    Tensor<float> output = layer.forward(input);
    cout << output.print() << endl;

    return 0;
}
