#include "DenseLayer.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {10, 6};
    int shape_weight[] = {2, 6};
    Tensor<float> input = Tensor<float>::getRandom(shape_input, 2);
    cout << "Input" << endl;
    cout << input.print() << endl;
    DenseLayer<float> layer = DenseLayer<float>(6, 2, ACTIVATION::Linear);
    cout << "weights" << endl;
    cout << layer.weights.print() << endl;
    cout << "Bias" << endl;
    cout << layer.bias.print() << endl;
    Tensor<float> output = layer.forward(input);
    cout << "Output" << endl;
    cout << output.print() << endl;

    return 0;
}
