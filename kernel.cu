#include "DenseLayer.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
  // int shape_input[] = {100, 600};
  // int shape_weight[] = {200, 600};
  // Tensor<float> input = Tensor<float>::getOnes(shape_input, 2);
  // input = input + 3;
  // cout << "Input" << endl;
  // cout << input.print() << endl;
  // DenseLayer<float> layer = DenseLayer<float>(shape_weight, 2, Initialization::RANDOMS);
  // cout << "weights" << endl;
  // cout << layer.weights.print() << endl;
  // cout << "Bias" << endl;
  // cout << layer.bias.print() << endl;
  // Tensor<float> output = layer.forward(input);
  // cout << "Output" << endl;
  // cout << output.print() << endl;
    int shape_input[] = {600};
    Tensor<float> input = Tensor<float>::getOnes(shape_input, 1);
    input = Tensor<float>::reduceSumLastAxis(input);
    cout << "Input" << endl;
    cout << input.print() << endl;

    return 0;
}
